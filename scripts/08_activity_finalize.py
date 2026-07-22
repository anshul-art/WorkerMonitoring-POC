"""Stage D finalize: GT-calibrated activity + full-video movement export.

- Track 16: ACTIVE/IDLE/UNKNOWN using confirmed idle window cand04
- Tracks 1,2,3,4: movement scores only (PENDING_GT) or UNKNOWN_SPARSE
- No invented percentiles for tracks without visual idle GT
"""

from __future__ import annotations

import json
from importlib.machinery import SourceFileLoader
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
m = SourceFileLoader("act", str(ROOT / "scripts" / "06_activity_classify.py")).load_module()

# Confirmed idle GT (human)
GT_IDLE = {
    16: [(8014, 8712)],  # cand04 phone
}
# Rejected "low movement" that were actually active/occlusion — for false-idle checks
GT_REJECT_ACTIVE = {
    1: [(15218, 15506), (12036, 12280)],  # cand03, cand05 sorting
}
GT_REJECT_OCCLUSION = {
    4: [(1624, 1896), (14594, 14882)],  # cand01, cand02
}


def build_track(pose: pd.DataFrame, tid: int) -> pd.DataFrame:
    g = pose[pose.track_id == tid].copy()
    if g.empty:
        return g
    g = m.per_sample_displacement(g)
    g = m.rolling_movement(g, window_s=1.0, min_valid_fraction=0.5)
    return g


def calibrate_from_idle(g: pd.DataFrame, windows: list[tuple[int, int]]) -> dict:
    scored = g[g.activity == "PENDING"]["rolling_movement"].dropna()
    idle_parts = []
    for f0, f1 in windows:
        part = g[
            (g.frame_idx >= f0)
            & (g.frame_idx <= f1)
            & (g.activity == "PENDING")
        ]["rolling_movement"].dropna()
        idle_parts.append(part)
    idle = pd.concat(idle_parts) if idle_parts else pd.Series(dtype=float)
    if idle.empty or scored.empty:
        return {"threshold": float("nan"), "method": "no_data", "idle": idle, "scored": scored}

    # Threshold: cover ~95% of confirmed idle, with small margin
    idle_p95 = float(np.quantile(idle, 0.95))
    thresh = idle_p95 * 1.05
    # Also report where this sits in overall distribution
    pct = float((scored < thresh).mean() * 100)
    return {
        "threshold": thresh,
        "method": "gt_idle_p95_x1.05",
        "idle": idle,
        "scored": scored,
        "idle_median": float(idle.median()),
        "idle_p95": idle_p95,
        "idle_max": float(idle.max()),
        "overall_pct_below_thresh": pct,
    }


def apply_labels(g: pd.DataFrame, thresh: float | None, has_gt: bool) -> pd.DataFrame:
    out = g.copy()
    labels = []
    for _, row in out.iterrows():
        if row["valid_fraction"] < 0.5 or pd.isna(row["rolling_movement"]):
            labels.append("UNKNOWN")
        elif not has_gt:
            labels.append("PENDING_GT")
        elif row["rolling_movement"] < thresh:
            labels.append("IDLE")
        else:
            labels.append("ACTIVE")
    out["activity"] = labels
    out["activity_threshold"] = thresh if has_gt else np.nan
    out["threshold_source"] = "gt_idle_cand04" if has_gt else "none_pending_gt"
    return out


def window_metrics(g: pd.DataFrame, f0: int, f1: int) -> dict:
    sub = g[(g.frame_idx >= f0) & (g.frame_idx <= f1)]
    if sub.empty:
        return {"n": 0}
    vc = sub["activity"].value_counts(normalize=True).to_dict()
    return {
        "n": len(sub),
        "pct_IDLE": float(vc.get("IDLE", 0) * 100),
        "pct_ACTIVE": float(vc.get("ACTIVE", 0) * 100),
        "pct_UNKNOWN": float(vc.get("UNKNOWN", 0) * 100),
        "pct_PENDING_GT": float(vc.get("PENDING_GT", 0) * 100),
        "med_roll": float(sub["rolling_movement"].median(skipna=True)),
    }


def main() -> int:
    cfg = m.load_cfg()
    pose = m.load_pose(ROOT / "outputs/pose/fasttracker_full/pose_landmarks.csv")
    pose = m.attach_scale(pose, ROOT / cfg["pose"]["tracks_csv"])
    out = ROOT / "outputs/activity/fasttracker_full"
    out.mkdir(parents=True, exist_ok=True)
    report_dir = out / "stage_d_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    real_ids = [1, 2, 3, 4, 16]
    calib_info = {}
    all_rows = []
    summary_rows = []

    print("=" * 60)
    print("Stage D finalize")
    print("=" * 60)

    for tid in real_ids:
        g = build_track(pose, tid)
        if g.empty:
            print(f"Skip track {tid}: empty")
            continue
        has_gt = tid in GT_IDLE
        thresh = None
        method = "none_pending_gt"
        if has_gt:
            cal = calibrate_from_idle(g, GT_IDLE[tid])
            thresh = cal["threshold"]
            method = cal["method"]
            calib_info[tid] = {k: v for k, v in cal.items() if k not in ("idle", "scored")}
            calib_info[tid]["n_idle_samples"] = int(len(cal["idle"]))
            calib_info[tid]["n_overall_scored"] = int(len(cal["scored"]))
            print(
                f"Track {tid}: GT thresh={thresh:.4f} ({method}) "
                f"idle_med={cal['idle_median']:.4f} overall_below={cal['overall_pct_below_thresh']:.1f}%"
            )
            # GT coverage plot
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.hist(cal["scored"], bins=40, color="#457b9d", alpha=0.85, edgecolor="white", label="overall")
            ax.hist(cal["idle"], bins=25, color="#e63946", alpha=0.7, edgecolor="white", label="GT idle cand04")
            ax.axvline(thresh, color="#c1121f", lw=2, label=f"IDLE thresh={thresh:.3f}")
            ax.set_title(f"Track {tid} — GT-calibrated IDLE threshold")
            ax.set_xlabel("rolling_movement")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(report_dir / f"track{tid}_gt_threshold.png", dpi=130)
            plt.close(fig)
        else:
            print(f"Track {tid}: no idle GT — PENDING_GT / UNKNOWN only")

        labeled = apply_labels(g, thresh, has_gt)
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
            "threshold_source",
        ]
        labeled[cols].to_csv(out / f"activity_full_track{tid}.csv", index=False)
        all_rows.append(labeled[cols])

        vc = labeled["activity"].value_counts()
        scored = labeled[labeled["rolling_movement"].notna()]
        summary_rows.append(
            {
                "track_id": tid,
                "station_id": int(labeled.station_id.mode().iloc[0]),
                "n_samples": len(labeled),
                "has_idle_gt": has_gt,
                "threshold": thresh if has_gt else None,
                "threshold_method": method,
                "n_ACTIVE": int(vc.get("ACTIVE", 0)),
                "n_IDLE": int(vc.get("IDLE", 0)),
                "n_UNKNOWN": int(vc.get("UNKNOWN", 0)),
                "n_PENDING_GT": int(vc.get("PENDING_GT", 0)),
                "pct_ACTIVE": float(vc.get("ACTIVE", 0) / len(labeled) * 100),
                "pct_IDLE": float(vc.get("IDLE", 0) / len(labeled) * 100),
                "pct_UNKNOWN": float(vc.get("UNKNOWN", 0) / len(labeled) * 100),
                "pct_PENDING_GT": float(vc.get("PENDING_GT", 0) / len(labeled) * 100),
                "med_rolling_movement": float(scored["rolling_movement"].median()) if len(scored) else None,
                "mean_valid_fraction": float(labeled["valid_fraction"].mean()),
            }
        )

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(out / "activity_full_all_tracks.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(report_dir / "per_track_summary.csv", index=False)

    # Hard-slice extract
    hard = combined[(combined.frame_idx >= 800) & (combined.frame_idx < 2200)].copy()
    hard.to_csv(out / "activity_hard_800_2200_all_tracks.csv", index=False)

    # Validation metrics on known windows
    val_rows = []
    g16 = combined[combined.track_id == 16]
    for f0, f1 in GT_IDLE[16]:
        met = window_metrics(g16, f0, f1)
        met.update({"window": "cand04_CONFIRM_idle", "track_id": 16, "f0": f0, "f1": f1})
        val_rows.append(met)
        print(f"Validation cand04 IDLE window: {met}")

    # Ambiguous cand06 — should not be mostly forced IDLE if threshold tight; report only
    met6 = window_metrics(g16, 9318, 9620)
    met6.update({"window": "cand06_AMBIGUOUS", "track_id": 16, "f0": 9318, "f1": 9620})
    val_rows.append(met6)

    g1 = combined[combined.track_id == 1]
    for f0, f1 in GT_REJECT_ACTIVE.get(1, []):
        met = window_metrics(g1, f0, f1)
        met.update({"window": "REJECT_was_sorting", "track_id": 1, "f0": f0, "f1": f1})
        val_rows.append(met)

    g4 = combined[combined.track_id == 4]
    for f0, f1 in GT_REJECT_OCCLUSION.get(4, []):
        met = window_metrics(g4, f0, f1)
        met.update({"window": "REJECT_occlusion", "track_id": 4, "f0": f0, "f1": f1})
        val_rows.append(met)

    # Mid absence: no rows for 3/16 in gap
    mid_gap = combined[
        (combined.track_id.isin([3, 16]))
        & (combined.frame_idx >= 1151)
        & (combined.frame_idx <= 1927)
    ]
    val_rows.append(
        {
            "window": "mid_absence_1151_1927",
            "track_id": "3+16",
            "f0": 1151,
            "f1": 1927,
            "n": len(mid_gap),
            "note": "expected 0 pose/activity rows",
        }
    )
    print(f"Mid absence pose rows 1151-1927: {len(mid_gap)} (expect 0)")

    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(report_dir / "validation_windows.csv", index=False)

    # Stacked activity timeline plot for track 16
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    t16 = combined[combined.track_id == 16].sort_values("frame_idx")
    axes[0].plot(t16.frame_idx, t16.rolling_movement, color="#457b9d", lw=0.6, alpha=0.8)
    axes[0].axhline(calib_info[16]["threshold"], color="#e63946", ls="--", label="IDLE thresh")
    axes[0].axvspan(8014, 8712, color="#e63946", alpha=0.15, label="GT idle cand04")
    axes[0].set_ylabel("rolling_movement")
    axes[0].set_title("Track 16 (mid) — movement + GT idle window")
    axes[0].legend(fontsize=8)

    color = {"ACTIVE": "#2a9d8f", "IDLE": "#e63946", "UNKNOWN": "#adb5bd", "PENDING_GT": "#f4a261"}
    for lab, sub in t16.groupby("activity"):
        axes[1].scatter(sub.frame_idx, [lab] * len(sub), s=2, c=color.get(lab, "#333"), label=lab)
    axes[1].set_xlabel("frame_idx")
    axes[1].set_title("Track 16 activity labels (full video)")
    axes[1].legend(fontsize=8, markerscale=3)
    fig.tight_layout()
    fig.savefig(report_dir / "track16_full_timeline.png", dpi=130)
    plt.close(fig)

    # Comparison bar chart: activity mix
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(summary))
    w = 0.2
    ax.bar(x - 1.5 * w, summary["pct_ACTIVE"], w, label="ACTIVE", color="#2a9d8f")
    ax.bar(x - 0.5 * w, summary["pct_IDLE"], w, label="IDLE", color="#e63946")
    ax.bar(x + 0.5 * w, summary["pct_UNKNOWN"], w, label="UNKNOWN", color="#adb5bd")
    ax.bar(x + 1.5 * w, summary["pct_PENDING_GT"], w, label="PENDING_GT", color="#f4a261")
    ax.set_xticks(x)
    ax.set_xticklabels([f"t{t}" for t in summary.track_id])
    ax.set_ylabel("% of pose samples")
    ax.set_title("Stage D activity mix by track (IDLE only where GT exists)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(report_dir / "activity_mix_by_track.png", dpi=130)
    plt.close(fig)

    # Write human report
    lines = []
    lines.append("STAGE D — Activity / Movement — FINAL OUTPUTS")
    lines.append("=" * 60)
    lines.append("")
    lines.append("POLICY")
    lines.append("- Scale: bbox diagonal; window: 1.0s; UNKNOWN if valid_fraction < 0.5")
    lines.append("- Null wrists never count as zero movement")
    lines.append("- ACTIVE/IDLE only for tracks with confirmed visual idle GT")
    lines.append("- Currently GT idle: track 16 only (cand04 phone, frames 8014-8712)")
    lines.append("- Tracks 1,2,3,4: PENDING_GT (scores present) or UNKNOWN — not labeled IDLE")
    lines.append("")
    lines.append("CALIBRATION (track 16)")
    if 16 in calib_info:
        c = calib_info[16]
        lines.append(f"- method: {c['method']}")
        lines.append(f"- threshold: {c['threshold']:.4f}")
        lines.append(f"- idle median / p95 / max: {c['idle_median']:.4f} / {c['idle_p95']:.4f} / {c['idle_max']:.4f}")
        lines.append(f"- % of track-16 overall below thresh: {c['overall_pct_below_thresh']:.1f}%")
    lines.append("")
    lines.append("PER-TRACK SUMMARY")
    lines.append(summary.to_string(index=False))
    lines.append("")
    lines.append("VALIDATION WINDOWS")
    lines.append(val_df.to_string(index=False))
    lines.append("")
    lines.append("KEY FILES")
    lines.append("- activity_full_all_tracks.csv")
    lines.append("- activity_full_track{1,2,3,4,16}.csv")
    lines.append("- activity_hard_800_2200_all_tracks.csv")
    lines.append("- stage_d_report/per_track_summary.csv")
    lines.append("- stage_d_report/validation_windows.csv")
    lines.append("- stage_d_report/track16_gt_threshold.png")
    lines.append("- stage_d_report/track16_full_timeline.png")
    lines.append("- stage_d_report/activity_mix_by_track.png")
    lines.append("- Proof idle image: outputs/sanity/idle_candidates/cand04_mid_f8363.jpg")
    lines.append("")
    lines.append("COMPARE METRICS (for slides)")
    lines.append("- Tracking: ByteTrack 24 IDs vs FastTracker 25 IDs; mid gap = real absence")
    lines.append("- Pose: 37020 rows; null L/R wrists ~14%/26%")
    lines.append("- Activity: only track 16 has IDLE%; others PENDING_GT until more GT")
    text = "\n".join(lines) + "\n"
    (report_dir / "STAGE_D_REPORT.txt").write_text(text, encoding="utf-8")
    with open(report_dir / "calibration.json", "w", encoding="utf-8") as f:
        json.dump(calib_info, f, indent=2)

    print()
    print(text)
    print(f"Wrote report dir: {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
