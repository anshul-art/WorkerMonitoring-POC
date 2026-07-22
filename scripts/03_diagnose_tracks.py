"""Step 5: track diagnostics - fragmentation, spatial noise, Type-B phantoms.

Checks:
  1. Unique track ID count vs expected workers
  2. Duration / span per ID (flag short / fragmented tracks)
  3. Where noise IDs cluster spatially (station-level)
  4. Type-B: track whose mean position is close to another simultaneously live track
  5. Resume seams (from resume_events.jsonl) - excluded from fragmentation blame

Does NOT apply min-box-area / overlap filters (deferred until Type-B proves harmful).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_resume_frames(run_name: str) -> list[int]:
    path = ROOT / "outputs" / "checkpoints" / run_name / "resume_events.jsonl"
    frames: list[int] = []
    if not path.exists():
        return frames
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            frames.append(int(ev["resume_from_frame"]))
    return frames


def per_track_stats(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    g = df.groupby("track_id").agg(
        n_frames=("frame_idx", "count"),
        frame_min=("frame_idx", "min"),
        frame_max=("frame_idx", "max"),
        mean_cx=("cx", "mean"),
        mean_cy=("cy", "mean"),
        mean_w=("w", "mean"),
        mean_h=("h", "mean"),
        mean_conf=("conf", "mean"),
        mean_area=("area", "mean"),
    )
    g["span_frames"] = g["frame_max"] - g["frame_min"] + 1
    g["duration_s"] = g["n_frames"] / fps
    g["span_s"] = g["span_frames"] / fps
    g["fill_ratio"] = g["n_frames"] / g["span_frames"]
    return g.sort_values("n_frames", ascending=False)


def classify_tracks(stats: pd.DataFrame, total_frames: int, fps: float) -> pd.DataFrame:
    """Label primary vs short/noise candidates by duration."""
    out = stats.copy()
    # Primary: covers a large fraction of the video
    primary_thresh = 0.5 * total_frames
    short_thresh_s = 3.0  # < 3 seconds of detections
    out["role"] = "secondary"
    out.loc[out["n_frames"] >= primary_thresh, "role"] = "primary"
    out.loc[out["duration_s"] < short_thresh_s, "role"] = "short_noise"
    # Medium fragments: not primary, but longer than short noise
    mid = (out["role"] == "secondary") & (out["n_frames"] < primary_thresh)
    out.loc[mid & (out["duration_s"] >= short_thresh_s), "role"] = "fragment"
    return out


def type_b_pairs(
    df: pd.DataFrame,
    stats: pd.DataFrame,
    dist_thresh_px: float = 80.0,
    min_overlap_frames: int = 10,
) -> pd.DataFrame:
    """Find track pairs that are simultaneously live and spatially close.

    Signature of Type-B double-box -> phantom ID: two IDs active in the same
    frames with centers persistently near each other.
    """
    ids = sorted(stats.index.tolist())
    frames_by_id = {tid: set(g["frame_idx"].tolist()) for tid, g in df.groupby("track_id")}
    rows: list[dict] = []

    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            overlap = frames_by_id[a] & frames_by_id[b]
            if len(overlap) < min_overlap_frames:
                continue
            sub = df[df["frame_idx"].isin(overlap) & df["track_id"].isin([a, b])]
            # pairwise distance per overlapping frame
            wide = sub.pivot_table(index="frame_idx", columns="track_id", values=["cx", "cy"])
            if a not in wide["cx"].columns or b not in wide["cx"].columns:
                continue
            dx = wide["cx"][a] - wide["cx"][b]
            dy = wide["cy"][a] - wide["cy"][b]
            dist = np.sqrt(dx**2 + dy**2).dropna()
            if len(dist) < min_overlap_frames:
                continue
            mean_dist = float(dist.mean())
            frac_close = float((dist < dist_thresh_px).mean())
            if mean_dist < dist_thresh_px or frac_close >= 0.5:
                rows.append(
                    {
                        "track_a": a,
                        "track_b": b,
                        "overlap_frames": len(dist),
                        "mean_dist_px": round(mean_dist, 1),
                        "frac_close": round(frac_close, 3),
                        "mean_cx_a": round(float(stats.loc[a, "mean_cx"]), 1),
                        "mean_cy_a": round(float(stats.loc[a, "mean_cy"]), 1),
                        "mean_cx_b": round(float(stats.loc[b, "mean_cx"]), 1),
                        "mean_cy_b": round(float(stats.loc[b, "mean_cy"]), 1),
                        "n_a": int(stats.loc[a, "n_frames"]),
                        "n_b": int(stats.loc[b, "n_frames"]),
                        "role_a": stats.loc[a, "role"],
                        "role_b": stats.loc[b, "role"],
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["frac_close", "overlap_frames"], ascending=[False, False]
    ) if rows else pd.DataFrame()


def nearest_primary(stats: pd.DataFrame) -> pd.Series:
    primaries = stats[stats["role"] == "primary"]
    if primaries.empty:
        # fallback: top-4 by duration
        primaries = stats.nlargest(4, "n_frames")

    def _nn(row):
        d = np.sqrt(
            (primaries["mean_cx"] - row["mean_cx"]) ** 2
            + (primaries["mean_cy"] - row["mean_cy"]) ** 2
        )
        tid = int(d.idxmin())
        return pd.Series({"nearest_primary": tid, "dist_to_primary_px": float(d.min())})

    return stats.apply(_nn, axis=1)


def make_plots(stats: pd.DataFrame, type_b: pd.DataFrame, out_dir: Path, expected: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Duration bar
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = {
        "primary": "#2a9d8f",
        "fragment": "#e9c46a",
        "short_noise": "#e76f51",
        "secondary": "#8d99ae",
    }
    order = stats.sort_values("n_frames", ascending=True)
    ax.barh(
        [str(i) for i in order.index],
        order["duration_s"],
        color=[colors.get(r, "#8d99ae") for r in order["role"]],
    )
    ax.set_xlabel("Duration (seconds of detections)")
    ax.set_ylabel("track_id")
    ax.set_title(f"Track durations - {len(stats)} IDs (expected {expected})")
    fig.tight_layout()
    fig.savefig(out_dir / "track_durations.png", dpi=120)
    plt.close(fig)

    # Spatial map
    fig, ax = plt.subplots(figsize=(9, 5))
    for role, marker, size in [
        ("primary", "o", 120),
        ("fragment", "s", 70),
        ("short_noise", "x", 50),
        ("secondary", "D", 60),
    ]:
        sub = stats[stats["role"] == role]
        if sub.empty:
            continue
        ax.scatter(
            sub["mean_cx"],
            sub["mean_cy"],
            c=colors[role],
            marker=marker,
            s=size,
            label=role,
            zorder=3,
        )
        for tid, row in sub.iterrows():
            ax.annotate(str(tid), (row["mean_cx"], row["mean_cy"]), fontsize=7, xytext=(4, 4), textcoords="offset points")

    if not type_b.empty:
        for _, row in type_b.head(12).iterrows():
            ax.plot(
                [row["mean_cx_a"], row["mean_cx_b"]],
                [row["mean_cy_a"], row["mean_cy_b"]],
                color="#c1121f",
                alpha=0.5,
                linewidth=1,
                zorder=2,
            )

    ax.invert_yaxis()  # image coords
    ax.set_xlabel("mean cx (px)")
    ax.set_ylabel("mean cy (px)")
    ax.set_title("Mean track positions (red lines = Type-B close pairs)")
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out_dir / "track_spatial_map.png", dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 5: diagnose tracks")
    parser.add_argument("--run-name", default="full")
    parser.add_argument("--dist-thresh", type=float, default=80.0, help="Type-B center distance (px)")
    parser.add_argument("--min-overlap", type=int, default=10, help="Min simultaneous frames for Type-B")
    args = parser.parse_args()

    cfg = load_config()
    expected = int(cfg["video"]["expected_workers"])
    fps = float(cfg["video"]["fps"])
    csv_path = ROOT / "outputs" / "tracks" / args.run_name / "tracks.csv"
    out_dir = ROOT / cfg["paths"]["diagnostics"] / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"ERROR: missing {csv_path}")
        return 1

    df = pd.read_csv(csv_path)
    # area helper column for deferred analysis reporting only
    df["area"] = df["w"] * df["h"]

    total_frames = int(df["frame_idx"].max()) + 1
    resume_frames = load_resume_frames(args.run_name)

    stats = per_track_stats(df, fps)
    stats = classify_tracks(stats, total_frames, fps)
    nn = nearest_primary(stats)
    stats = pd.concat([stats, nn], axis=1)

    type_b = type_b_pairs(df, stats, dist_thresh_px=args.dist_thresh, min_overlap_frames=args.min_overlap)

    # Per-frame occupancy
    occ = df.groupby("frame_idx").size()
    n_lt = int((occ < expected).sum())
    n_eq = int((occ == expected).sum())
    n_gt = int((occ > expected).sum())

    # Save tables
    stats.to_csv(out_dir / "track_stats.csv")
    if not type_b.empty:
        type_b.to_csv(out_dir / "type_b_pairs.csv", index=False)
    else:
        (out_dir / "type_b_pairs.csv").write_text("track_a,track_b,overlap_frames,mean_dist_px,frac_close\n", encoding="utf-8")

    noise = stats[stats["role"].isin(["short_noise", "fragment"])].copy()
    if not noise.empty:
        cluster = (
            noise.groupby("nearest_primary")
            .agg(
                n_noise_ids=("n_frames", "count"),
                total_noise_frames=("n_frames", "sum"),
                mean_dist=("dist_to_primary_px", "mean"),
            )
            .sort_values("total_noise_frames", ascending=False)
        )
        cluster.to_csv(out_dir / "noise_by_station.csv")
    else:
        cluster = pd.DataFrame()

    make_plots(stats, type_b, out_dir, expected)

    # Console report
    print("=" * 60)
    print("Step 5 - Track diagnostics")
    print("=" * 60)
    print(f"CSV: {csv_path}")
    print(f"Frames: {total_frames}  |  Unique IDs: {len(stats)}  |  Expected workers: {expected}")
    print(f"Resume seams: {resume_frames if resume_frames else 'none (clean single session)'}")
    print()
    print("Occupancy (tracks/frame):")
    print(f"  <{expected}: {n_lt} frames ({100 * n_lt / total_frames:.1f}%)")
    print(f"  ={expected}: {n_eq} frames ({100 * n_eq / total_frames:.1f}%)")
    print(f"  >{expected}: {n_gt} frames ({100 * n_gt / total_frames:.1f}%)")
    print()
    print("Track roles:")
    print(stats["role"].value_counts().to_string())
    print()
    print("Primary / long tracks:")
    cols = ["n_frames", "duration_s", "mean_cx", "mean_cy", "mean_conf", "role", "nearest_primary", "dist_to_primary_px"]
    print(stats[cols].head(12).to_string())
    print()

    if cluster.empty:
        print("Noise clustering: no short/fragment tracks.")
    else:
        print("Noise IDs clustered by nearest primary station:")
        print(cluster.to_string())
        print()
        # Left-worker station: primary with smallest mean_cx among primaries
        primaries = stats[stats["role"] == "primary"]
        if not primaries.empty:
            left_id = int(primaries["mean_cx"].idxmin())
            left_noise = noise[noise["nearest_primary"] == left_id]
            print(f"Left-most primary station = track {left_id} (cx~={stats.loc[left_id, 'mean_cx']:.0f})")
            print(f"  Noise/fragment IDs near it: {sorted(left_noise.index.tolist())}")
            print(f"  Their total detection frames: {int(left_noise['n_frames'].sum()) if len(left_noise) else 0}")

    print()
    print(f"Type-B pairs (simultaneous + mean center < {args.dist_thresh}px or >=50% frames close):")
    serious = pd.DataFrame()
    if type_b.empty:
        print("  NONE found - Type-B double-box phantoms not evidenced at this threshold.")
        print("  Deferred filters (min-box-area / overlap) stay OFF.")
    else:
        print(type_b.head(15).to_string(index=False))
        # Primary/long track paired with a short/fragment = actionable Type-B
        serious = type_b[
            (
                (type_b["role_a"] == "primary")
                & (type_b["role_b"].isin(["short_noise", "fragment"]))
            )
            | (
                (type_b["role_b"] == "primary")
                & (type_b["role_a"].isin(["short_noise", "fragment"]))
            )
        ]
        print()
        if serious.empty:
            print("  No Type-B pair ties a primary worker to a noise/fragment track.")
            print("  Deferred filters stay OFF for now.")
        else:
            print(
                f"  {len(serious)} Type-B pair(s) involve a primary + noise/fragment - "
                "review before enabling filters."
            )
            print(serious.head(10).to_string(index=False))

    # Fragmentation narrative: ID handoff at same station
    print()
    print("Likely ID handoffs (same station, sequential fragments):")
    primaries = stats[stats["role"] == "primary"]
    for pid, prow in primaries.iterrows():
        near = stats[(stats["nearest_primary"] == pid) & (stats.index != pid)].sort_values("frame_min")
        if near.empty:
            continue
        print(f"  Station primary {pid} (cx~={prow['mean_cx']:.0f}, cy~={prow['mean_cy']:.0f}):")
        for tid, row in near.iterrows():
            print(
                f"    id={tid} role={row['role']} frames={int(row['frame_min'])}-{int(row['frame_max'])} "
                f"n={int(row['n_frames'])} dist={row['dist_to_primary_px']:.0f}px"
            )

    print()
    print(f"Outputs written to: {out_dir}")
    print("  track_stats.csv, type_b_pairs.csv, noise_by_station.csv")
    print("  track_durations.png, track_spatial_map.png")
    print()
    print("DECISION INPUT:")
    print(
        f"  unique_ids={len(stats)} vs expected={expected} -> "
        f"fragmentation={'YES' if len(stats) > expected else 'NO'}"
    )
    if not serious.empty:
        print("  Type-B primary involvement -> consider filters AFTER reviewing pairs")
    else:
        print("  Type-B not driving primary phantoms -> prefer track_buffer / match_thresh tune first")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
