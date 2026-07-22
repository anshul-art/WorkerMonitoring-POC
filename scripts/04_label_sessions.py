"""Step B: labeling-only diagnostic for FastTracker (NO stitching/merging).

Labels each raw track_id as real_session or noise based on duration distribution.
Groups real_session IDs by station for REPORTING only — does not modify tracks.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def choose_noise_threshold(n_frames: pd.Series, fps: float) -> tuple[int, str]:
    """Pick noise cutoff from duration distribution.

    Prefer the largest gap that separates short tracks from sessions lasting
    hundreds+ of frames (matches 'real_session' intent).
    """
    vals = np.sort(n_frames.to_numpy())
    best_cut = None
    best_gap = -1
    for i in range(len(vals) - 1):
        lo, hi = int(vals[i]), int(vals[i + 1])
        gap = hi - lo
        # Meaningful split: below is short (<~few seconds to ~few tens of s),
        # above is a real session (hundreds+ frames).
        if lo < 400 and hi >= 500 and gap > best_gap:
            best_gap = gap
            best_cut = lo  # noise if n_frames <= lo? use < hi with cut at lo
            # noise = n_frames <= lo  OR  n_frames < midpoint?
            # Use: noise if n_frames <= lo (everything at or below the lower side of the gap)
            # Actually label uses n_frames < threshold, so threshold = lo + 1
            best_cut = lo + 1

    if best_cut is not None:
        return best_cut, (
            f"largest short-vs-long gap in duration distribution "
            f"(cut at {best_cut} frames / {best_cut / fps:.1f}s; gap size {best_gap} frames)"
        )

    # Fallback: ~100 frames
    return 100, "fallback ~100 frames (~3.3s) — no clear hundreds-scale gap found"


def assign_stations_by_x(real: pd.DataFrame, expected: int) -> pd.DataFrame:
    """Informational left-to-right station grouping by mean_x (NOT a merge).

    Strategy: sort real sessions by mean_x, place station boundaries at the
    (expected-1) largest x-gaps so co-located workers (e.g. two right-side
    seats) stay separated when their x-gap is among the top splits.
    """
    out = real.copy()
    if out.empty:
        out["station_id"] = pd.Series(dtype=int)
        out["station_name"] = pd.Series(dtype=str)
        return out

    ordered = out.sort_values("mean_x")
    xs = ordered["mean_x"].to_numpy()
    idxs = list(ordered.index)

    if len(xs) == 1:
        station_of = {idxs[0]: 0}
    else:
        gaps = np.diff(xs)
        n_bounds = min(expected - 1, len(gaps))
        # Always take the largest n_bounds gaps (no 80px filter) so nearby
        # stations like foreground-right vs background-right still split.
        bound_idxs = set(int(i) for i in np.argsort(gaps)[-n_bounds:]) if n_bounds else set()

        station_of = {idxs[0]: 0}
        sid = 0
        for i in range(1, len(idxs)):
            if (i - 1) in bound_idxs:
                sid += 1
            station_of[idxs[i]] = sid

    out["station_id"] = out.index.map(station_of)
    order_ids = out.groupby("station_id")["mean_x"].mean().sort_values().index.tolist()
    remap = {old: new for new, old in enumerate(order_ids)}
    out["station_id"] = out["station_id"].map(remap)

    def _name(i: int) -> str:
        n = int(out["station_id"].max()) + 1 if len(out) else 1
        if i == 0:
            return "Station 1 (leftmost)"
        if i == n - 1 and n > 1:
            return f"Station {i + 1} (rightmost)"
        return f"Station {i + 1}"

    out["station_name"] = out["station_id"].map(_name)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Label FastTracker IDs (no stitching)")
    parser.add_argument("--run-name", default="fasttracker_full")
    parser.add_argument("--expected-stations", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config()
    fps = float(cfg["video"]["fps"])
    csv_path = ROOT / "outputs" / "tracks" / args.run_name / "tracks.csv"
    out_dir = ROOT / "outputs" / "diagnostics" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"ERROR: missing {csv_path}")
        return 1

    df = pd.read_csv(csv_path)
    required = {"frame_idx", "track_id", "x1", "y1", "x2", "y2", "cx", "cy"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: CSV missing columns {missing}")
        return 1

    named = dict(
        first_frame=("frame_idx", "min"),
        last_frame=("frame_idx", "max"),
        n_frames=("frame_idx", "count"),
        mean_x=("cx", "mean"),
        mean_y=("cy", "mean"),
    )
    if "conf" in df.columns:
        named["mean_conf"] = ("conf", "mean")

    stats = df.groupby("track_id").agg(**named).sort_values("n_frames", ascending=False)
    stats["duration_s"] = stats["n_frames"] / fps
    stats["span_frames"] = stats["last_frame"] - stats["first_frame"] + 1

    print("=" * 60)
    print("Step B — duration distribution (before labeling)")
    print("=" * 60)
    print(stats["n_frames"].describe(percentiles=[0.5, 0.75, 0.9, 0.95]).to_string())
    print()
    print("n_frames per track_id (sorted):")
    print(stats["n_frames"].to_string())

    noise_max_frames, reason = choose_noise_threshold(stats["n_frames"], fps)
    print()
    print(f"Noise threshold chosen: n_frames < {noise_max_frames} ({reason})")

    stats["label"] = np.where(stats["n_frames"] < noise_max_frames, "noise", "real_session")
    real = stats[stats["label"] == "real_session"].copy()
    noise_df = stats[stats["label"] == "noise"].copy()

    real = assign_stations_by_x(real, args.expected_stations)
    stats = stats.join(real[["station_id", "station_name"]], how="left")

    report_cols = [
        "first_frame",
        "last_frame",
        "n_frames",
        "duration_s",
        "mean_x",
        "mean_y",
        "label",
        "station_id",
        "station_name",
    ]
    out_csv = out_dir / "track_labels.csv"
    stats[report_cols].to_csv(out_csv)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(stats["n_frames"], bins=30, color="#457b9d", edgecolor="white")
    ax.axvline(
        noise_max_frames,
        color="#e63946",
        linestyle="--",
        label=f"noise < {noise_max_frames}",
    )
    ax.set_xlabel("n_frames")
    ax.set_ylabel("count of track_ids")
    ax.set_title("Track duration distribution (FastTracker raw IDs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "duration_distribution.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    if len(real):
        for sid, sub in real.groupby("station_id"):
            ax.scatter(sub["mean_x"], sub["mean_y"], s=80, label=f"station {int(sid) + 1}")
            for tid, row in sub.iterrows():
                ax.annotate(
                    str(tid),
                    (row["mean_x"], row["mean_y"]),
                    fontsize=8,
                    xytext=(4, 4),
                    textcoords="offset points",
                )
    if len(noise_df):
        ax.scatter(
            noise_df["mean_x"],
            noise_df["mean_y"],
            c="#adb5bd",
            marker="x",
            s=40,
            label="noise",
        )
    ax.invert_yaxis()
    ax.set_xlabel("mean_x (cx)")
    ax.set_ylabel("mean_y (cy)")
    ax.set_title("Real sessions by station (informational only — IDs not merged)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "real_sessions_spatial.png", dpi=120)
    plt.close(fig)

    lines: list[str] = []
    lines.append("FastTracker labeling report (NO stitching / NO CSV modification)")
    lines.append(f"Source tracks: {csv_path}")
    lines.append(f"Labels file: {out_csv}")
    lines.append(
        "station_id/station_name are REPORTING ONLY — track_ids were NOT merged."
    )
    lines.append("")
    lines.append(f"Total unique track_ids: {len(stats)}")
    lines.append(f"  real_session: {len(real)}")
    lines.append(f"  noise: {len(noise_df)}")
    lines.append(f"Noise rule: n_frames < {noise_max_frames} ({reason})")
    lines.append("")
    lines.append("Per-track table:")
    lines.append(
        stats[report_cols]
        .sort_values(["label", "n_frames"], ascending=[True, False])
        .to_string()
    )
    lines.append("")
    lines.append("Per-station narrative (informational only):")
    if len(real):
        for sid in sorted(real["station_id"].unique()):
            sub = real[real["station_id"] == sid].sort_values("first_frame")
            name = sub["station_name"].iloc[0]
            mx = sub["mean_x"].mean()
            lines.append(
                f"  {name} (mean x~={mx:.0f}): {len(sub)} real_session track_id(s)"
            )
            for tid, row in sub.iterrows():
                lines.append(
                    f"    track_id={tid}: frames {int(row['first_frame'])}-{int(row['last_frame'])} "
                    f"(n={int(row['n_frames'])}, {row['duration_s']:.1f}s)"
                )
            if len(sub) >= 2:
                prev_last = None
                for tid, row in sub.sort_values("first_frame").iterrows():
                    if prev_last is not None:
                        gap = int(row["first_frame"]) - prev_last - 1
                        if gap > 0:
                            lines.append(
                                f"    -> gap of {gap} frames (~{gap / fps:.1f}s) before "
                                f"track_id={tid} (same station, SEPARATE IDs — not merged)"
                            )
                    prev_last = int(row["last_frame"])
    lines.append("")
    lines.append(
        "CONFIRMATION: tracks.csv was not modified. No track_ids were merged or reassigned."
    )

    narrative_path = out_dir / "session_narrative.txt"
    narrative_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print("\n".join(lines))
    print()
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {narrative_path}")
    print(f"Wrote: {out_dir / 'duration_distribution.png'}")
    print(f"Wrote: {out_dir / 'real_sessions_spatial.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
