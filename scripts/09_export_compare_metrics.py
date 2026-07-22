"""Export a single comparison pack for cross-folder / cross-pipeline compare.

Writes under outputs/activity/<run>/compare_pack/:
  - COMPARE_METRICS.json
  - COMPARE_METRICS.csv  (flat key,value)
  - per_track_movement_percentiles.csv
  - gt_window_scores.csv
  - README_COMPARE.md

Usage:
  python scripts/09_export_compare_metrics.py
  python scripts/09_export_compare_metrics.py --run-name fasttracker_full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Known GT windows (for score tables; idle fit only uses CONFIRM)
GT_WINDOWS = [
    {"cand": "cand04_CONFIRM_idle", "track_id": 16, "f0": 8014, "f1": 8712, "decision": "CONFIRM"},
    {"cand": "cand06_PARTIAL", "track_id": 16, "f0": 9318, "f1": 9620, "decision": "PARTIAL"},
    {"cand": "t1_low01", "track_id": 1, "f0": 10420, "f1": 10780, "decision": "REJECT"},
    {"cand": "t1_low02", "track_id": 1, "f0": 4052, "f1": 4412, "decision": "REJECT"},
    {"cand": "t1_low03", "track_id": 1, "f0": 10, "f1": 310, "decision": "REJECT"},
    {"cand": "t1_low04", "track_id": 1, "f0": 12298, "f1": 12634, "decision": "REJECT"},
    {"cand": "t1_low05", "track_id": 1, "f0": 1264, "f1": 1624, "decision": "REJECT"},
    {"cand": "t2_low01", "track_id": 2, "f0": 128, "f1": 488, "decision": "REJECT"},
    {"cand": "t2_low02", "track_id": 2, "f0": 4426, "f1": 4786, "decision": "REJECT"},
    {"cand": "t2_low03", "track_id": 2, "f0": 16706, "f1": 17066, "decision": "REJECT"},
    {"cand": "t2_low04", "track_id": 2, "f0": 10840, "f1": 11200, "decision": "PARTIAL"},
    {"cand": "t2_low05", "track_id": 2, "f0": 12882, "f1": 13242, "decision": "PARTIAL"},
]


def pct(s: pd.Series, q: float) -> float:
    s = s.dropna()
    if s.empty:
        return float("nan")
    return float(np.quantile(s, q))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="fasttracker_full")
    args = ap.parse_args()

    act_dir = ROOT / "outputs" / "activity" / args.run_name
    tracks_csv = ROOT / "outputs" / "tracks" / args.run_name / "tracks.csv"
    pose_csv = ROOT / "outputs" / "pose" / args.run_name / "pose_landmarks.csv"
    summary_csv = act_dir / "stage_d_report" / "per_track_summary.csv"
    calib_json = act_dir / "stage_d_report" / "calibration.json"
    out = act_dir / "compare_pack"
    out.mkdir(parents=True, exist_ok=True)

    act = pd.read_csv(act_dir / "activity_full_all_tracks.csv")
    tracks = pd.read_csv(tracks_csv) if tracks_csv.exists() else None
    pose = pd.read_csv(pose_csv) if pose_csv.exists() else None
    summary = pd.read_csv(summary_csv) if summary_csv.exists() else None
    calib = json.loads(calib_json.read_text()) if calib_json.exists() else {}

    # --- per-track movement percentiles (GT-independent; best for folder compare) ---
    rows = []
    for tid, g in act.groupby("track_id"):
        scored = g["rolling_movement"].dropna()
        vf = g["valid_fraction"].dropna()
        label_counts = g["activity"].value_counts().to_dict()
        rows.append(
            {
                "track_id": int(tid),
                "n_samples": int(len(g)),
                "n_scored": int(scored.shape[0]),
                "mean_valid_fraction": float(vf.mean()) if len(vf) else float("nan"),
                "roll_p05": pct(scored, 0.05),
                "roll_p25": pct(scored, 0.25),
                "roll_p50": pct(scored, 0.50),
                "roll_p75": pct(scored, 0.75),
                "roll_p95": pct(scored, 0.95),
                "roll_mean": float(scored.mean()) if len(scored) else float("nan"),
                "pct_ACTIVE": 100.0 * label_counts.get("ACTIVE", 0) / max(len(g), 1),
                "pct_IDLE": 100.0 * label_counts.get("IDLE", 0) / max(len(g), 1),
                "pct_UNKNOWN": 100.0 * label_counts.get("UNKNOWN", 0) / max(len(g), 1),
                "pct_PENDING_GT": 100.0 * label_counts.get("PENDING_GT", 0) / max(len(g), 1),
            }
        )
    perc_df = pd.DataFrame(rows).sort_values("track_id")
    perc_df.to_csv(out / "per_track_movement_percentiles.csv", index=False)

    # --- GT window scores ---
    gw_rows = []
    for w in GT_WINDOWS:
        g = act[
            (act.track_id == w["track_id"])
            & (act.frame_idx >= w["f0"])
            & (act.frame_idx <= w["f1"])
        ]
        scored = g["rolling_movement"].dropna()
        gw_rows.append(
            {
                **w,
                "n": int(len(g)),
                "n_scored": int(len(scored)),
                "med_roll": float(scored.median()) if len(scored) else float("nan"),
                "mean_roll": float(scored.mean()) if len(scored) else float("nan"),
                "p95_roll": pct(scored, 0.95),
                "mean_vf": float(g["valid_fraction"].mean()) if len(g) else float("nan"),
            }
        )
    gw_df = pd.DataFrame(gw_rows)
    gw_df.to_csv(out / "gt_window_scores.csv", index=False)

    # --- headline metrics ---
    metrics: dict = {
        "run_name": args.run_name,
        "video_assumed": "20260602_214750.mp4",
        "n_activity_rows": int(len(act)),
        "track_ids": sorted(int(x) for x in act.track_id.unique()),
        "idle_gt_tracks": sorted(int(k) for k in calib.keys()),
        "calibration": calib,
    }

    if tracks is not None:
        tid_col = "track_id" if "track_id" in tracks.columns else "id"
        metrics["n_unique_track_ids"] = int(tracks[tid_col].nunique())
        metrics["n_track_rows"] = int(len(tracks))
        if "frame_idx" in tracks.columns:
            metrics["frame_min"] = int(tracks.frame_idx.min())
            metrics["frame_max"] = int(tracks.frame_idx.max())

    if pose is not None:
        metrics["n_pose_rows"] = int(len(pose))
        for side in ("left", "right"):
            vcol = f"{side}_wrist_vis"
            if vcol in pose.columns:
                metrics[f"mean_{side}_wrist_vis"] = float(pose[vcol].mean())

    # Mid absence blackout (tracks 3+16 empty)
    mid = act[act.track_id.isin([3, 16])]
    if not mid.empty:
        present = set(mid.frame_idx.unique())
        fmin, fmax = int(act.frame_idx.min()), int(act.frame_idx.max())
        missing = [f for f in range(fmin, fmax + 1, 2) if f not in present and 1120 <= f <= 1970]
        metrics["mid_band_missing_samples_approx_1120_1970"] = len(missing)

    # Hard slice quick stats
    hard = act[(act.frame_idx >= 800) & (act.frame_idx <= 2200)]
    metrics["hard_800_2200_n_rows"] = int(len(hard))
    metrics["hard_800_2200_med_roll_all"] = float(hard["rolling_movement"].median())

    if summary is not None:
        metrics["per_track_summary"] = summary.to_dict(orient="records")

    # Confirmed idle vs reject contrast (for slides)
    conf = gw_df[gw_df.decision == "CONFIRM"]
    rej = gw_df[gw_df.decision == "REJECT"]
    metrics["confirm_idle_med_roll_mean"] = float(conf.med_roll.mean()) if len(conf) else None
    metrics["reject_working_med_roll_mean"] = float(rej.med_roll.mean()) if len(rej) else None

    (out / "COMPARE_METRICS.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    flat = []
    for k, v in metrics.items():
        if isinstance(v, (dict, list)):
            flat.append({"key": k, "value": json.dumps(v)})
        else:
            flat.append({"key": k, "value": v})
    pd.DataFrame(flat).to_csv(out / "COMPARE_METRICS.csv", index=False)

    readme = f"""# Compare pack — `{args.run_name}`

Use these files to compare against another pipeline folder on the **same video**.

## Best apples-to-apples columns
From `per_track_movement_percentiles.csv`:
- `roll_p05`, `roll_p50`, `roll_p95`, `mean_valid_fraction`, `n_samples`
- `pct_UNKNOWN` (occlusion / sparse pose)
- `pct_IDLE` / `pct_ACTIVE` only meaningful where idle GT exists (here: track **16**)

From `gt_window_scores.csv`:
- Same frame windows → compare `med_roll` / `p95_roll` across folders

From `COMPARE_METRICS.json`:
- `n_unique_track_ids`, calibration threshold, mid-gap sample count

## Idle labeling status (this run)
- Track 16 only has ACTIVE/IDLE (cand04 phone GT)
- Tracks 1 & 2: all low-movement candidates REJECT or PARTIAL (pick/drop = working) → **PENDING_GT**
- Do not compare IDLE% on tracks 1/2/4 until both folders share CONFIRM idle GT

## Quick copy checklist
1. Copy entire `compare_pack/` from each folder
2. Diff `per_track_movement_percentiles.csv` on track_id
3. Diff `gt_window_scores.csv` on cand
4. Only then compare IDLE% on track 16
"""
    (out / "README_COMPARE.md").write_text(readme, encoding="utf-8")
    print(f"Wrote {out}")
    print(perc_df.to_string(index=False))
    print()
    print(gw_df[["cand", "track_id", "decision", "med_roll", "p95_roll", "mean_vf"]].to_string(index=False))


if __name__ == "__main__":
    main()
