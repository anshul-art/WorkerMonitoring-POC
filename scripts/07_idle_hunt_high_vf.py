"""Pull lowest-movement, high-valid_fraction windows for tracks 1 and 2."""

from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
m = SourceFileLoader("act", str(ROOT / "scripts" / "06_activity_classify.py")).load_module()

MIN_VF = 0.8
MIN_DUR_S = 8.0
MAX_DUR_S = 20.0
TOP_N = 5


def main() -> int:
    cfg = m.load_cfg()
    pose = m.load_pose(ROOT / "outputs/pose/fasttracker_full/pose_landmarks.csv")
    pose = m.attach_scale(pose, ROOT / cfg["pose"]["tracks_csv"])

    out = ROOT / "outputs/sanity/idle_candidates_t1_t2"
    out.mkdir(parents=True, exist_ok=True)

    rows_meta: list[dict] = []
    cap = cv2.VideoCapture(str(ROOT / "20260602_214750.mp4"))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    for tid in [1, 2]:
        g = pose[pose.track_id == tid].copy()
        g = m.per_sample_displacement(g)
        g = m.rolling_movement(g, window_s=1.0, min_valid_fraction=0.5)
        ok = g[
            (g.valid_fraction > MIN_VF)
            & g.rolling_movement.notna()
            & (g.activity == "PENDING")
        ].copy()
        print(f"\nTrack {tid}: high-vf scored samples={len(ok)} / {len(g)}")
        if ok.empty:
            continue

        ok = ok.sort_values("rolling_movement")
        used_ranges: list[tuple[int, int]] = []
        windows: list[dict] = []

        for _, row in ok.iterrows():
            center = int(row.frame_idx)
            half = int(12 * fps / 2)
            f0, f1 = center - half, center + half
            seg = ok[(ok.frame_idx >= f0) & (ok.frame_idx <= f1)]
            if len(seg) < 20:
                continue
            f0, f1 = int(seg.frame_idx.min()), int(seg.frame_idx.max())
            dur = (f1 - f0) / fps
            if dur < MIN_DUR_S:
                near = ok[
                    (ok.frame_idx >= center - int(10 * fps))
                    & (ok.frame_idx <= center + int(10 * fps))
                ]
                if near.empty:
                    continue
                f0, f1 = int(near.frame_idx.min()), int(near.frame_idx.max())
                dur = (f1 - f0) / fps
            if dur > MAX_DUR_S:
                mid = (f0 + f1) // 2
                f0 = mid - int(MAX_DUR_S * fps / 2)
                f1 = mid + int(MAX_DUR_S * fps / 2)
                dur = (f1 - f0) / fps

            if any(not (f1 < a or f0 > b) for a, b in used_ranges):
                continue

            seg2 = ok[(ok.frame_idx >= f0) & (ok.frame_idx <= f1)]
            if seg2.empty:
                continue
            win_all = g[(g.frame_idx >= f0) & (g.frame_idx <= f1)]
            mean_vf = float(win_all["valid_fraction"].mean())
            if mean_vf < MIN_VF:
                continue

            med = float(seg2.rolling_movement.median())
            windows.append(
                {
                    "track_id": tid,
                    "station_id": int(g.station_id.mode().iloc[0]),
                    "f0": f0,
                    "f1": f1,
                    "dur_s": round(dur, 1),
                    "med_roll": med,
                    "mean_vf": round(mean_vf, 3),
                    "n_scored": len(seg2),
                }
            )
            used_ranges.append((f0, f1))
            if len(windows) >= TOP_N:
                break

        windows = sorted(windows, key=lambda w: w["med_roll"])
        print(f"  selected {len(windows)} windows")
        for i, w in enumerate(windows, 1):
            t0 = w["f0"] / fps
            t1 = w["f1"] / fps
            print(
                f"  #{i} frames {w['f0']}-{w['f1']} ({t0:.1f}-{t1:.1f}s) "
                f"dur={w['dur_s']}s med={w['med_roll']:.4f} vf={w['mean_vf']}"
            )
            idxs = np.linspace(w["f0"], w["f1"], 6).astype(int)
            tiles = []
            for idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                okf, fr = cap.read()
                if not okf:
                    continue
                fr = cv2.resize(fr, (426, 240))
                cv2.putText(
                    fr,
                    f"f{idx}",
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                )
                tiles.append(fr)
            if not tiles:
                continue
            strip = np.hstack(tiles)
            tag = f"t{tid}_low{i:02d}_f{w['f0']}-{w['f1']}"
            strip_name = f"{tag}_strip.jpg"
            cv2.imwrite(str(out / strip_name), strip, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            mid = (w["f0"] + w["f1"]) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            okf, fr = cap.read()
            mid_name = f"{tag}_mid_f{mid}.jpg"
            if okf:
                cv2.imwrite(str(out / mid_name), fr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            w["cand"] = f"t{tid}_low{i:02d}"
            w["strip"] = strip_name
            w["mid"] = mid_name
            w["t0_s"] = round(w["f0"] / fps, 1)
            w["t1_s"] = round(w["f1"] / fps, 1)
            rows_meta.append(w)

    cap.release()
    meta = pd.DataFrame(rows_meta)
    meta.to_csv(out / "index.csv", index=False)
    (out / "REVIEW_ME.md").write_text(
        """# Track 1 & 2 — lowest movement, HIGH valid_fraction (>0.8)

These are hunt pointers for visual idle confirmation (NOT labels yet).
Filter: mean valid_fraction in window > 0.8 (genuine low motion, not sparse wrists).

## What to review
For each row in `index.csv`, open:
- `*_strip.jpg` — 6 frames across the window
- `*_mid_f*.jpg` — full-res middle frame

Focus on the named track:
- **track 1** = foreground-right (near camera, sorting bin)
- **track 2** = leftmost worker

## Decide per window
CONFIRM idle / REJECT (still sorting) / PARTIAL

Reply e.g.: `t2_low01 CONFIRM, t1_low02 REJECT, ...`
""",
        encoding="utf-8",
    )
    print(f"\nWrote {out}")
    if len(meta):
        print(meta[["cand", "track_id", "f0", "f1", "t0_s", "t1_s", "dur_s", "med_roll", "mean_vf"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
