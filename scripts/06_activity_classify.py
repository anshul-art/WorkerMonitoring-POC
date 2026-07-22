"""Stage 4: scale-normalized wrist movement + ACTIVE/IDLE/UNKNOWN.

Pipeline:
  1) Per-frame wrist displacement, normalized by bbox diagonal (from tracks.csv)
  2) Rolling sum over ~1s window (video time)
  3) Per-track threshold calibration from movement histogram
  4) Classify: ACTIVE / IDLE / UNKNOWN (valid_fraction < min_valid_fraction)

Null policy (locked):
  - Null wrist coords never count as zero motion
  - Only valid observations enter displacement
  - Sparse windows -> UNKNOWN
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]


def load_cfg() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_xy(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("", np.nan), errors="coerce")


def load_pose(pose_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(pose_csv, keep_default_na=False)
    for col in [
        "l_wrist_x",
        "l_wrist_y",
        "r_wrist_x",
        "r_wrist_y",
        "l_wrist_vis",
        "r_wrist_vis",
    ]:
        df[col] = parse_xy(df[col])
    return df


def attach_scale(pose: pd.DataFrame, tracks_csv: Path) -> pd.DataFrame:
    """bbox diagonal as per-frame scale reference."""
    tr = pd.read_csv(tracks_csv)[["frame_idx", "track_id", "w", "h"]]
    tr["scale"] = np.sqrt(tr["w"] ** 2 + tr["h"] ** 2)
    tr["scale"] = tr["scale"].replace(0, np.nan)
    out = pose.merge(tr, on=["frame_idx", "track_id"], how="left")
    return out


def per_sample_displacement(g: pd.DataFrame) -> pd.DataFrame:
    """Compute scale-normalized wrist step between consecutive pose samples.

    Uses whichever wrists are valid on BOTH consecutive samples.
    If both L and R valid on a step: average of the two normalized distances.
    If only one: that one. If neither: NaN (does not become 0).
    """
    g = g.sort_values("frame_idx").reset_index(drop=True)
    out = g.copy()

    def step_norm(x0, y0, x1, y1, scale) -> float:
        if any(pd.isna(v) for v in (x0, y0, x1, y1, scale)) or scale <= 0:
            return np.nan
        return float(np.hypot(x1 - x0, y1 - y0) / scale)

    n = len(g)
    disp = np.full(n, np.nan)
    used = np.zeros(n, dtype=object)

    for i in range(1, n):
        scale = g.loc[i, "scale"]  # normalize by current frame scale
        parts = []
        tags = []
        # left
        dL = step_norm(
            g.loc[i - 1, "l_wrist_x"],
            g.loc[i - 1, "l_wrist_y"],
            g.loc[i, "l_wrist_x"],
            g.loc[i, "l_wrist_y"],
            scale,
        )
        if not np.isnan(dL):
            parts.append(dL)
            tags.append("L")
        # right
        dR = step_norm(
            g.loc[i - 1, "r_wrist_x"],
            g.loc[i - 1, "r_wrist_y"],
            g.loc[i, "r_wrist_x"],
            g.loc[i, "r_wrist_y"],
            scale,
        )
        if not np.isnan(dR):
            parts.append(dR)
            tags.append("R")

        if parts:
            disp[i] = float(np.mean(parts))
            used[i] = "+".join(tags)
        else:
            disp[i] = np.nan
            used[i] = ""

    out["disp_norm"] = disp
    out["disp_wrists"] = used
    out["sample_valid"] = ~np.isnan(disp)
    # first row has no previous -> not a valid movement sample
    out.loc[0, "sample_valid"] = False
    return out


def rolling_movement(
    g: pd.DataFrame,
    window_s: float,
    min_valid_fraction: float,
) -> pd.DataFrame:
    """Trailing time window in seconds over pose timestamps."""
    g = g.sort_values("timestamp_s").reset_index(drop=True)
    ts = g["timestamp_s"].to_numpy()
    disp = g["disp_norm"].to_numpy()
    valid = g["sample_valid"].to_numpy()

    n = len(g)
    roll = np.full(n, np.nan)
    frac = np.zeros(n)
    n_valid = np.zeros(n, dtype=int)
    n_slots = np.zeros(n, dtype=int)
    label = np.array(["UNKNOWN"] * n, dtype=object)

    j = 0
    for i in range(n):
        t_end = ts[i]
        t_start = t_end - window_s
        while j < i and ts[j] < t_start:
            j += 1
        # window samples (j..i inclusive), excluding the invalid first-of-track steps
        sl = slice(j, i + 1)
        slot_valid = valid[sl]
        n_slots[i] = int(slot_valid.size)
        n_valid[i] = int(slot_valid.sum())
        frac[i] = (n_valid[i] / n_slots[i]) if n_slots[i] else 0.0

        if frac[i] < min_valid_fraction or n_valid[i] == 0:
            roll[i] = np.nan
            label[i] = "UNKNOWN"
        else:
            # sum only valid displacements in window (nulls excluded, not zero)
            vals = disp[sl][slot_valid]
            roll[i] = float(np.nansum(vals))
            label[i] = "PENDING"  # threshold applied later

    g = g.copy()
    g["rolling_movement"] = roll
    g["valid_fraction"] = frac
    g["n_valid_in_window"] = n_valid
    g["n_slots_in_window"] = n_slots
    g["activity"] = label
    return g


def assess_bimodality(values: np.ndarray, bins: int = 40) -> dict:
    """Histogram + simple valley check between two peaks."""
    values = values[np.isfinite(values)]
    if len(values) < 50:
        return {
            "n": len(values),
            "bimodal": False,
            "reason": "too few samples",
            "counts": None,
            "edges": None,
            "peaks": [],
            "valley": None,
        }

    counts, edges = np.histogram(values, bins=bins)
    # smooth lightly
    kern = np.array([1, 2, 3, 2, 1], dtype=float)
    kern /= kern.sum()
    smooth = np.convolve(counts.astype(float), kern, mode="same")
    peaks, props = find_peaks(smooth, prominence=max(1.0, smooth.max() * 0.05))
    peaks = sorted(peaks.tolist(), key=lambda p: smooth[p], reverse=True)

    result = {
        "n": len(values),
        "counts": counts,
        "edges": edges,
        "smooth": smooth,
        "peaks": peaks[:4],
        "peak_centers": [float(0.5 * (edges[p] + edges[p + 1])) for p in peaks[:4]],
        "bimodal": False,
        "valley": None,
        "reason": "",
    }

    if len(peaks) < 2:
        result["reason"] = f"found {len(peaks)} peak(s) — not bimodal"
        return result

    p1, p2 = sorted(peaks[:2])
    if p2 - p1 < 2:
        result["reason"] = "two peaks too close"
        return result

    valley_idx = int(p1 + np.argmin(smooth[p1 : p2 + 1]))
    valley_val = float(0.5 * (edges[valley_idx] + edges[min(valley_idx + 1, len(edges) - 1)]))
    # require valley meaningfully below both peaks
    peak_h = min(smooth[p1], smooth[p2])
    if smooth[valley_idx] > peak_h * 0.85:
        result["reason"] = "no clear valley between peaks"
        return result

    result["bimodal"] = True
    result["valley"] = valley_val
    result["reason"] = "clear valley between two peaks"
    return result


def calibrate_threshold(values: np.ndarray, assessment: dict) -> tuple[float, str]:
    """Per-track ACTIVE threshold. Prefer valley if bimodal; else report fallback."""
    values = values[np.isfinite(values)]
    if assessment.get("bimodal") and assessment.get("valley") is not None:
        return float(assessment["valley"]), "bimodal_valley"

    # Not bimodal — do NOT invent a confident threshold. Use median as provisional
    # split for exploration only, clearly marked.
    if len(values) == 0:
        return float("nan"), "no_data"
    med = float(np.median(values))
    return med, "provisional_median_NOT_bimodal"


def apply_threshold(g: pd.DataFrame, thresh: float) -> pd.DataFrame:
    out = g.copy()
    mask_pending = out["activity"] == "PENDING"
    out.loc[mask_pending & (out["rolling_movement"] >= thresh), "activity"] = "ACTIVE"
    out.loc[mask_pending & (out["rolling_movement"] < thresh), "activity"] = "IDLE"
    # anything still PENDING with nan roll stays UNKNOWN
    out.loc[out["activity"] == "PENDING", "activity"] = "UNKNOWN"
    out["activity_threshold"] = thresh
    return out


def plot_histogram(
    values: np.ndarray,
    assessment: dict,
    thresh: float,
    thresh_method: str,
    title: str,
    out_path: Path,
) -> None:
    values = values[np.isfinite(values)]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(values, bins=40, color="#457b9d", edgecolor="white", alpha=0.9)
    if assessment.get("peak_centers"):
        for c in assessment["peak_centers"][:2]:
            ax.axvline(c, color="#2a9d8f", linestyle=":", linewidth=1.5, label=f"peak~{c:.3f}")
    if assessment.get("valley") is not None:
        ax.axvline(
            assessment["valley"],
            color="#e9c46a",
            linestyle="--",
            linewidth=2,
            label=f"valley={assessment['valley']:.3f}",
        )
    ax.axvline(
        thresh,
        color="#e63946",
        linestyle="-",
        linewidth=2,
        label=f"threshold={thresh:.3f} ({thresh_method})",
    )
    ax.set_xlabel("rolling_movement (scale-normalized, ~1s window)")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def run_track(
    pose: pd.DataFrame,
    track_id: int,
    window_s: float,
    min_valid_fraction: float,
    frame_start: int | None,
    frame_end: int | None,
    out_dir: Path,
    tag: str,
) -> dict:
    g = pose[pose["track_id"] == track_id].copy()
    if frame_start is not None:
        g = g[g["frame_idx"] >= frame_start]
    if frame_end is not None:
        g = g[g["frame_idx"] < frame_end]
    if g.empty:
        print(f"No pose rows for track {track_id} in range")
        return {}

    g = per_sample_displacement(g)
    g = rolling_movement(g, window_s=window_s, min_valid_fraction=min_valid_fraction)

    # Calibrate on NON-UNKNOWN windows only
    calib_vals = g.loc[g["activity"] == "PENDING", "rolling_movement"].to_numpy()
    assessment = assess_bimodality(calib_vals)
    thresh, method = calibrate_threshold(calib_vals, assessment)
    g = apply_threshold(g, thresh)

    station = int(g["station_id"].mode().iloc[0]) if len(g) else -1
    title = (
        f"Track {track_id} (station {station}) — {tag}\n"
        f"bimodal={assessment['bimodal']} ({assessment['reason']})"
    )
    plot_histogram(
        calib_vals,
        assessment,
        thresh,
        method,
        title,
        out_dir / f"hist_track{track_id}_{tag}.png",
    )

    # summary
    counts = g["activity"].value_counts().to_dict()
    print("=" * 60)
    print(f"Track {track_id} | station {station} | {tag}")
    print(f"Pose samples: {len(g)}  frames {int(g.frame_idx.min())}-{int(g.frame_idx.max())}")
    print(
        f"Scale: bbox diagonal | window={window_s}s | "
        f"min_valid_fraction={min_valid_fraction}"
    )
    print(f"Valid step rate: {g['sample_valid'].mean():.1%}")
    print(f"Bimodal: {assessment['bimodal']} — {assessment['reason']}")
    if assessment.get("peak_centers"):
        print(f"Peak centers: {assessment['peak_centers'][:2]}")
    print(f"Threshold: {thresh:.4f} ({method})")
    print("Activity counts:", counts)
    if len(calib_vals):
        print(
            f"rolling_movement stats (classifiable): "
            f"min={np.nanmin(calib_vals):.4f} med={np.nanmedian(calib_vals):.4f} "
            f"max={np.nanmax(calib_vals):.4f}"
        )

    out_csv = out_dir / f"activity_track{track_id}_{tag}.csv"
    cols = [
        "frame_idx",
        "timestamp_s",
        "track_id",
        "station_id",
        "scale",
        "disp_norm",
        "sample_valid",
        "rolling_movement",
        "valid_fraction",
        "activity",
        "activity_threshold",
    ]
    g[cols].to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_dir / f'hist_track{track_id}_{tag}.png'}")

    return {
        "track_id": track_id,
        "station_id": station,
        "threshold": thresh,
        "method": method,
        "bimodal": assessment["bimodal"],
        "reason": assessment["reason"],
        "counts": counts,
        "df": g,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Activity classification Stage 4")
    parser.add_argument("--track-id", type=int, default=None, help="Single track test")
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--tag", default="test")
    parser.add_argument("--all-real", action="store_true", help="All real track_ids in range")
    parser.add_argument("--window-s", type=float, default=1.0)
    args = parser.parse_args()

    cfg = load_cfg()
    pose_cfg = cfg["pose"]
    mov = pose_cfg.get("movement", {})
    min_frac = float(mov.get("min_valid_fraction", 0.5))

    pose = load_pose(ROOT / "outputs/pose/fasttracker_full/pose_landmarks.csv")
    pose = attach_scale(pose, ROOT / pose_cfg["tracks_csv"])

    out_dir = ROOT / "outputs" / "activity" / "fasttracker_full"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all_real:
        ids = list(pose_cfg["real_track_ids"])
    elif args.track_id is not None:
        ids = [args.track_id]
    else:
        print("Specify --track-id or --all-real")
        return 1

    results = []
    for tid in ids:
        # skip tracks with no rows in range
        sub = pose[pose["track_id"] == tid]
        if args.frame_start is not None:
            sub = sub[sub["frame_idx"] >= args.frame_start]
        if args.frame_end is not None:
            sub = sub[sub["frame_idx"] < args.frame_end]
        if sub.empty:
            print(f"Skip track {tid}: no pose in range")
            continue
        results.append(
            run_track(
                pose,
                tid,
                window_s=args.window_s,
                min_valid_fraction=min_frac,
                frame_start=args.frame_start,
                frame_end=args.frame_end,
                out_dir=out_dir,
                tag=args.tag,
            )
        )

    # summary table
    if results:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for r in results:
            if not r:
                continue
            print(
                f"  track {r['track_id']} station {r['station_id']}: "
                f"bimodal={r['bimodal']} thresh={r['threshold']:.4f} ({r['method']}) "
                f"counts={r['counts']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
