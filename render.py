#!/usr/bin/env python3
"""Render charts/*.svg, data/summary.json and the README numbers table from data/*.csv.

Usage: uv run render.py

Output is deterministic (fixed SVG hash salt, no timestamps, glyphs as paths with
the bundled DejaVu Sans), so re-rendering unchanged data produces no git diff.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CHARTS = ROOT / "charts"
README = ROOT / "README.md"
START, END = "<!-- stats:start -->", "<!-- stats:end -->"
MIN_DAYS_FOR_DOWNLOADS_CHART = 14

# Light chart card; reads fine embedded on both GitHub light and dark pages.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BORDER = "#e1e0d9"
BLUE = "#2a78d6"
BLUE_LIGHT = "#9ec5f4"

plt.rcParams.update(
    {
        "svg.hashsalt": "gptme-stats",
        "svg.fonttype": "path",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.75,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "legend.frameon": False,
    }
)

comma = FuncFormatter(lambda v, _: f"{v:,.0f}")


def read_csv(name: str) -> list[dict[str, str]]:
    path = DATA / name
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def new_figure(title: str, subtitle: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.subplots_adjust(left=0.09, right=0.86, top=0.8, bottom=0.12)
    fig.patch.set_edgecolor(BORDER)
    fig.patch.set_linewidth(1.5)
    fig.text(0.02, 0.93, title, fontsize=14, fontweight="bold", color=INK)
    fig.text(0.02, 0.865, subtitle, fontsize=9.5, color=INK_2)
    ax.tick_params(axis="both", length=0, pad=6)
    ax.tick_params(axis="x", length=4, color=BASELINE)
    ax.yaxis.set_major_formatter(comma)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, min_n_ticks=3))
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, show_offset=False))
    return fig, ax


def end_label(ax, x, y, text: str) -> None:
    ax.plot([x], [y], "o", ms=8, color=BLUE, mec=SURFACE, mew=2, zorder=5, clip_on=False)
    ax.annotate(
        text,
        (x, y),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=10,
        fontweight="bold",
        color=INK,
        annotation_clip=False,
    )


def save(fig, name: str) -> None:
    CHARTS.mkdir(exist_ok=True)
    fig.savefig(CHARTS / name, format="svg", metadata={"Date": None, "Creator": None})
    plt.close(fig)


def render_stars(stars: list[str], daily: list[dict[str, str]], as_of: str) -> None:
    per_day = Counter(s[:10] for s in stars)
    dates, totals, running = [], [], 0
    for day in sorted(per_day):
        running += per_day[day]
        dates.append(dt.date.fromisoformat(day))
        totals.append(running)
    # Per-star timestamps need a user token; when they stop updating, continue the
    # curve from the daily stargazers_count snapshots.
    for row in sorted(daily, key=lambda r: r["date"]):
        day = dt.date.fromisoformat(row["date"])
        if day > dates[-1] and row.get("stars", "").isdigit():
            dates.append(day)
            totals.append(int(row["stars"]))
    fig, ax = new_figure("gptme GitHub stars", f"Cumulative stargazers of gptme/gptme, as of {as_of}")
    ax.fill_between(dates, totals, color=BLUE, alpha=0.1, linewidth=0, step="post")
    ax.step(dates, totals, where="post", color=BLUE, linewidth=2)
    ax.set_ylim(0, max(totals) * 1.08)
    ax.set_xlim(dates[0], dates[-1])
    end_label(ax, dates[-1], totals[-1], f"{totals[-1]:,}")
    save(fig, "stars.svg")


def rolling_mean(dates: list[dt.date], values: dict[dt.date, int], window: int = 30):
    xs, ys = [], []
    first = dates[0]
    for d in dates:
        if (d - first).days < window - 1:
            continue
        present = [values[d - dt.timedelta(days=i)] for i in range(window) if d - dt.timedelta(days=i) in values]
        if len(present) >= window * 0.8:
            xs.append(d)
            ys.append(sum(present) / len(present))
    return xs, ys


def render_pypi(pypi: list[dict[str, str]]) -> None:
    values = {dt.date.fromisoformat(r["date"]): int(r["downloads"]) for r in pypi}
    dates = sorted(values)
    fig, ax = new_figure(
        "gptme PyPI downloads",
        f"Daily downloads of the gptme package (pypistats, without mirrors), {dates[0]} to {dates[-1]}",
    )
    ax.bar(dates, [values[d] for d in dates], width=0.8, color=BLUE_LIGHT, linewidth=0, label="Daily")
    xs, ys = rolling_mean(dates, values)
    if xs:
        ax.plot(xs, ys, color=BLUE, linewidth=2, label="30-day average")
        end_label(ax, xs[-1], ys[-1], f"{ys[-1]:,.0f}/day")
    ax.set_xlim(dates[0] - dt.timedelta(days=1), dates[-1] + dt.timedelta(days=1))
    ax.set_ylim(0, max(values.values()) * 1.08)
    ax.legend(loc="upper left", fontsize=9, labelcolor=INK_2, handlelength=1.5)
    save(fig, "pypi.svg")


def render_downloads(daily: list[dict[str, str]]) -> bool:
    rows = [r for r in daily if r.get("release_downloads")]
    if len(rows) < MIN_DAYS_FOR_DOWNLOADS_CHART:
        return False
    dates = [dt.date.fromisoformat(r["date"]) for r in rows]
    totals = [int(r["release_downloads"]) for r in rows]
    fig, ax = new_figure(
        "gptme release downloads",
        "Cumulative GitHub release asset downloads, all releases of gptme/gptme",
    )
    ax.plot(dates, totals, color=BLUE, linewidth=2)
    ax.set_xlim(dates[0], dates[-1])
    ax.set_ylim(0, max(totals) * 1.08)
    end_label(ax, dates[-1], totals[-1], f"{totals[-1]:,}")
    save(fig, "downloads.svg")
    return True


def pypi_window(pypi: list[dict[str, str]], days: int) -> int:
    return sum(int(r["downloads"]) for r in pypi[-days:])


def build_summary(daily, stars, pypi) -> dict:
    latest = daily[-1]
    as_of = dt.date.fromisoformat(latest["date"])
    cutoff = (as_of - dt.timedelta(days=30)).isoformat()

    def num(key: str) -> int | None:
        return int(latest[key]) if latest.get(key) else None

    return {
        "generated_at": latest["date"],
        "source": "https://github.com/gptme/stats",
        "repo": "gptme/gptme",
        "stars": num("stars"),
        "stars_last_30d": sum(1 for s in stars if s[:10] > cutoff),
        "forks": num("forks"),
        "watchers": num("watchers"),
        "open_issues": num("open_issues"),
        "open_prs": num("open_prs"),
        "contributors": num("contributors"),
        "release_downloads": num("release_downloads"),
        "tauri_release_downloads": num("tauri_release_downloads"),
        "pypi": {
            "package": "gptme",
            "last_day": {"date": pypi[-1]["date"], "downloads": int(pypi[-1]["downloads"])},
            "last_7d": pypi_window(pypi, 7),
            "last_30d": pypi_window(pypi, 30),
            "tracked_total": pypi_window(pypi, len(pypi)),
            "tracked_since": pypi[0]["date"],
        },
    }


def fmt(v: int | None) -> str:
    return "n/a" if v is None else f"{v:,}"


def update_readme(s: dict, has_downloads_chart: bool) -> None:
    p = s["pypi"]
    lines = [
        START,
        f"_Latest snapshot: {s['generated_at']} (UTC). Machine-readable: [`data/summary.json`](data/summary.json)._",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| GitHub stars | {fmt(s['stars'])} |",
        f"| Stars gained, last 30 days | {fmt(s['stars_last_30d'])} |",
        f"| Forks | {fmt(s['forks'])} |",
        f"| Watchers | {fmt(s['watchers'])} |",
        f"| Contributors | {fmt(s['contributors'])} |",
        f"| Open issues | {fmt(s['open_issues'])} |",
        f"| Open pull requests | {fmt(s['open_prs'])} |",
        f"| Release asset downloads (gptme/gptme) | {fmt(s['release_downloads'])} |",
        f"| Release asset downloads (gptme/gptme-tauri) | {fmt(s['tauri_release_downloads'])} |",
        f"| PyPI downloads, {p['last_day']['date']} | {fmt(p['last_day']['downloads'])} |",
        f"| PyPI downloads, last 7 days | {fmt(p['last_7d'])} |",
        f"| PyPI downloads, last 30 days | {fmt(p['last_30d'])} |",
        f"| PyPI downloads tracked since {p['tracked_since']} | {fmt(p['tracked_total'])} |",
    ]
    if has_downloads_chart:
        lines += ["", "![GitHub release downloads](charts/downloads.svg)"]
    lines.append(END)
    text = README.read_text()
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
    if not pattern.search(text):
        raise SystemExit(f"README.md is missing the {START} ... {END} markers")
    README.write_text(pattern.sub(lambda _: "\n".join(lines), text))


def main() -> None:
    daily = read_csv("daily.csv")
    stars = [r["starred_at"] for r in read_csv("stars.csv")]
    pypi = read_csv("pypi_daily.csv")
    if not (daily and stars and pypi):
        raise SystemExit("missing data, run collect.py first")

    summary = build_summary(daily, stars, pypi)
    render_stars(stars, daily, summary["generated_at"])
    render_pypi(pypi)
    has_downloads = render_downloads(daily)
    (DATA / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    update_readme(summary, has_downloads)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
