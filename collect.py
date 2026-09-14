#!/usr/bin/env python3
"""Collect gptme community and usage stats into data/*.csv.

Usage: uv run collect.py [--full-stars]

All network requests run first. Files are written (atomically) only after
every source has been fetched and validated, so a failed run never leaves a
partial or garbage row behind.

Needs no secrets beyond an optional GITHUB_TOKEN (or GH_TOKEN), which only
raises the GitHub API rate limit.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
STARS_CSV = DATA / "stars.csv"
DAILY_CSV = DATA / "daily.csv"
PYPI_CSV = DATA / "pypi_daily.csv"

REPO = "gptme/gptme"
TAURI_REPO = "gptme/gptme-tauri"
PYPI_PACKAGE = "gptme"

GH_API = "https://api.github.com"
PYPISTATS_URL = f"https://pypistats.org/api/packages/{PYPI_PACKAGE}/overall"
STAR_ACCEPT = "application/vnd.github.star+json"
PER_PAGE = 100

DAILY_FIELDS = [
    "date",
    "stars",
    "forks",
    "watchers",
    "open_issues",
    "open_prs",
    "contributors",
    "release_downloads",
    "tauri_release_downloads",
]

ISO_TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class TokenNotAccepted(RuntimeError):
    """401, or 403 "Resource not accessible by integration".

    Listing stargazers with timestamps requires a user token: the workflow's
    repo-scoped GITHUB_TOKEN gets a 403 and unauthenticated requests get a 401.
    """


class Http:
    """requests wrapper with retries for 429, 5xx, network errors and GitHub rate limits."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "gptme-stats (+https://github.com/gptme/stats)"
        self.gh_headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        token = (
            os.environ.get("STATS_GH_TOKEN")
            or os.environ.get("GITHUB_TOKEN")
            or os.environ.get("GH_TOKEN")
        )
        if token:
            self.gh_headers["Authorization"] = f"Bearer {token}"
        else:
            log("warning: no GITHUB_TOKEN set, using unauthenticated GitHub API (60 req/h)")

    def get(
        self,
        url: str,
        *,
        params: dict | None = None,
        headers: dict | None = None,
        attempts: int = 6,
        allow_404: bool = False,
    ) -> requests.Response:
        last_error = ""
        for attempt in range(attempts):
            resp: requests.Response | None = None
            try:
                resp = self.session.get(url, params=params, headers=headers, timeout=30)
            except requests.RequestException as e:
                last_error = f"{type(e).__name__}: {e}"
            else:
                if resp.status_code < 400 or (allow_404 and resp.status_code == 404):
                    return resp
                rate_limited = resp.status_code == 429 or (
                    resp.status_code == 403
                    and (
                        resp.headers.get("x-ratelimit-remaining") == "0"
                        or "rate limit" in resp.text.lower()
                    )
                )
                if resp.status_code == 401 or (
                    resp.status_code == 403 and "not accessible by integration" in resp.text
                ):
                    raise TokenNotAccepted(
                        f"GET {resp.url} -> HTTP {resp.status_code}: {resp.text[:300]}"
                    )
                if not (rate_limited or resp.status_code >= 500):
                    raise RuntimeError(
                        f"GET {resp.url} -> HTTP {resp.status_code}: {resp.text[:300]}"
                    )
                last_error = f"HTTP {resp.status_code}"

            if attempt == attempts - 1:
                break
            delay = float(min(2 ** (attempt + 1), 60))
            if resp is not None:
                retry_after = resp.headers.get("retry-after", "")
                reset = resp.headers.get("x-ratelimit-reset", "")
                if retry_after.isdigit():
                    delay = max(delay, float(retry_after))
                elif resp.headers.get("x-ratelimit-remaining") == "0" and reset.isdigit():
                    delay = max(delay, int(reset) - time.time() + 2)
            if delay > 900:
                raise RuntimeError(f"GET {url}: rate limited, reset in {delay:.0f}s, giving up")
            log(f"GET {url} failed ({last_error}), retry {attempt + 1} in {delay:.0f}s")
            time.sleep(delay)
        raise RuntimeError(f"GET {url} failed after {attempts} attempts: {last_error}")

    def gh(
        self,
        path: str,
        params: dict | None = None,
        accept: str | None = None,
        allow_404: bool = False,
    ) -> requests.Response:
        headers = dict(self.gh_headers)
        if accept:
            headers["Accept"] = accept
        return self.get(f"{GH_API}{path}", params=params, headers=headers, allow_404=allow_404)


def expect_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name}: expected non-negative int, got {value!r}")
    return value


def count_via_link(http: Http, path: str, params: dict | None = None) -> int:
    """Count items of a paginated GitHub list endpoint with a single request (per_page=1)."""
    resp = http.gh(path, params={**(params or {}), "per_page": 1})
    last = resp.links.get("last", {}).get("url")
    if last:
        m = re.search(r"[?&]page=(\d+)", last)
        if not m:
            raise ValueError(f"{path}: cannot parse last page from {last}")
        return int(m.group(1))
    items = resp.json()
    if not isinstance(items, list):
        raise ValueError(f"{path}: expected list, got {type(items).__name__}")
    return len(items)


# --- stars -------------------------------------------------------------------


def fetch_star_page(http: Http, page: int) -> tuple[list[str], requests.Response]:
    resp = http.gh(
        f"/repos/{REPO}/stargazers",
        params={"per_page": PER_PAGE, "page": page},
        accept=STAR_ACCEPT,
    )
    items = resp.json()
    if not isinstance(items, list):
        raise ValueError(f"stargazers page {page}: expected list")
    stamps = []
    for item in items:
        ts = item.get("starred_at") if isinstance(item, dict) else None
        if not isinstance(ts, str) or not ISO_TS.match(ts):
            raise ValueError(f"stargazers page {page}: bad starred_at {ts!r}")
        stamps.append(ts)
    return stamps, resp


def fetch_all_stars(http: Http) -> list[str]:
    stamps: list[str] = []
    page = 1
    while True:
        batch, resp = fetch_star_page(http, page)
        stamps.extend(batch)
        if "next" not in resp.links or not batch:
            break
        page += 1
    log(f"stars: full fetch, {page} pages, {len(stamps)} stargazers")
    return sorted(stamps)


def merge_new_stars(stored: list[str], fetched: list[str]) -> list[str]:
    """Append stars from `fetched` that are newer than the stored list.

    `fetched` must contain every current stargazer with starred_at >= stored[-1]
    (the stargazers API lists oldest first, so the tail pages suffice).
    Timestamps have one-second resolution, so ties on the last stored second
    are resolved by count.
    """
    if not stored:
        return sorted(fetched)
    last = stored[-1]
    stored_ties = stored.count(last)
    fetched_ties = sum(1 for t in fetched if t == last)
    newer = sorted(t for t in fetched if t > last)
    return stored + [last] * max(0, fetched_ties - stored_ties) + newer


def fetch_stars_incremental(http: Http, stored: list[str], total: int) -> list[str] | None:
    """Fetch only the tail pages. Returns None if an incremental update isn't possible."""
    if not stored:
        return None
    last = stored[-1]
    page = max(1, math.ceil(total / PER_PAGE))
    fetched: list[str] = []
    pages = 0
    while page >= 1:
        batch, _ = fetch_star_page(http, page)
        pages += 1
        fetched = batch + fetched
        # The page starts strictly before our last known star: everything newer is covered.
        if batch and batch[0] < last:
            break
        page -= 1
        if pages > 5:
            return None  # far behind or heavy churn: cheaper and safer to refetch everything
    log(f"stars: incremental fetch, {pages} page(s)")
    return merge_new_stars(stored, fetched)


def collect_stars(http: Http, stored: list[str], total: int, force_full: bool) -> list[str]:
    try:
        return _collect_stars(http, stored, total, force_full)
    except TokenNotAccepted as e:
        # Per-star timestamps need a user token (STATS_GH_TOKEN secret). Without one, keep
        # the stored history; daily.csv still records the stargazers_count for the chart.
        log(f"warning: star timestamps unavailable, keeping stars.csv as is ({e})")
        return stored


def _collect_stars(http: Http, stored: list[str], total: int, force_full: bool) -> list[str]:
    stars = None if force_full else fetch_stars_incremental(http, stored, total)
    if stars is not None and len(stars) != total:
        # Unstars (or stars racing the count) make the stored list diverge; resync from scratch.
        log(f"stars: incremental result {len(stars)} != stargazers_count {total}, full resync")
        stars = None
    if stars is None:
        stars = fetch_all_stars(http)
    if stored and len(stars) < 0.9 * len(stored):
        raise ValueError(f"stars: refusing to shrink from {len(stored)} to {len(stars)} rows")
    return stars


# --- snapshot ----------------------------------------------------------------


def release_downloads(http: Http, repo: str) -> int:
    total = 0
    page = 1
    while True:
        resp = http.gh(f"/repos/{repo}/releases", params={"per_page": PER_PAGE, "page": page})
        releases = resp.json()
        if not isinstance(releases, list):
            raise ValueError(f"{repo} releases: expected list")
        for rel in releases:
            for asset in rel.get("assets", []):
                total += expect_int(asset.get("download_count"), f"{repo} asset download_count")
        if "next" not in resp.links or not releases:
            return total
        page += 1


def collect_snapshot(http: Http, today: str) -> tuple[dict[str, str], int]:
    repo = http.gh(f"/repos/{REPO}").json()
    stars = expect_int(repo.get("stargazers_count"), "stargazers_count")
    open_issues_and_prs = expect_int(repo.get("open_issues_count"), "open_issues_count")
    open_prs = count_via_link(http, f"/repos/{REPO}/pulls", {"state": "open"})

    tauri: int | None = None
    tauri_resp = http.gh(f"/repos/{TAURI_REPO}", allow_404=True)
    if tauri_resp.status_code == 200 and not tauri_resp.json().get("private", True):
        tauri = release_downloads(http, TAURI_REPO)

    row = {
        "date": today,
        "stars": stars,
        "forks": expect_int(repo.get("forks_count"), "forks_count"),
        "watchers": expect_int(repo.get("subscribers_count"), "subscribers_count"),
        "open_issues": max(0, open_issues_and_prs - open_prs),
        "open_prs": open_prs,
        "contributors": count_via_link(http, f"/repos/{REPO}/contributors"),
        "release_downloads": release_downloads(http, REPO),
        "tauri_release_downloads": "" if tauri is None else tauri,
    }
    return {k: str(v) for k, v in row.items()}, stars


# --- pypi --------------------------------------------------------------------


def collect_pypi(http: Http, today: str) -> dict[str, int]:
    resp = http.get(PYPISTATS_URL, params={"mirrors": "false"})
    payload = resp.json()
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("pypistats: empty or malformed response")
    out: dict[str, int] = {}
    for r in rows:
        if r.get("category") != "without_mirrors":
            continue
        date = r.get("date")
        if not isinstance(date, str) or not ISO_DATE.match(date):
            raise ValueError(f"pypistats: bad date {date!r}")
        if date >= today:
            continue  # today's number is still accumulating
        out[date] = expect_int(r.get("downloads"), f"pypistats downloads {date}")
    if not out:
        raise ValueError("pypistats: no without_mirrors rows")
    return out


# --- io ----------------------------------------------------------------------


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full-stars", action="store_true", help="refetch all stargazers")
    args = parser.parse_args()

    today = dt.datetime.now(dt.UTC).date().isoformat()
    http = Http()

    stored_stars = [r["starred_at"] for r in read_csv(STARS_CSV)]
    daily_rows = read_csv(DAILY_CSV)
    pypi_rows = {r["date"]: r["downloads"] for r in read_csv(PYPI_CSV)}

    # Fetch everything before writing anything.
    snapshot, star_total = collect_snapshot(http, today)
    stars = collect_stars(http, stored_stars, star_total, args.full_stars)
    pypi = collect_pypi(http, today)

    new_pypi_days = sorted(d for d in pypi if d not in pypi_rows)
    for d in new_pypi_days:
        pypi_rows[d] = str(pypi[d])
    daily_by_date = {r["date"]: r for r in daily_rows}
    daily_by_date[today] = snapshot  # re-running on the same UTC day overwrites that day only

    write_csv(STARS_CSV, ["starred_at"], [{"starred_at": s} for s in stars])
    write_csv(PYPI_CSV, ["date", "downloads"], [{"date": d, "downloads": pypi_rows[d]} for d in sorted(pypi_rows)])
    write_csv(DAILY_CSV, DAILY_FIELDS, [daily_by_date[d] for d in sorted(daily_by_date)])

    log(
        f"ok {today}: stars={snapshot['stars']} (stars.csv {len(stars)} rows, "
        f"+{len(stars) - len(stored_stars)}), forks={snapshot['forks']}, "
        f"contributors={snapshot['contributors']}, release_downloads={snapshot['release_downloads']}, "
        f"pypi +{len(new_pypi_days)} day(s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
