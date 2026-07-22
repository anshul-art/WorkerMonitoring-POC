"""Extract MediaPipe pose landmarks for real-session track_ids.

Contract (locked):
  - Every landmark includes a visibility score (*_vis).
  - If visibility < threshold: *_x / *_y are blank (null). Never write a
    low-confidence coordinate that could be read as a real position.
  - Downstream rolling_movement must:
      (1) exclude nulls from displacement (null != zero movement)
      (2) if valid_fraction in window < min_valid_fraction -> UNKNOWN
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

# MediaPipe Pose landmark indices
LM = {
    "l_wrist": 15,
    "r_wrist": 16,
    "l_elbow": 13,
    "r_elbow": 14,
}

CSV_HEADER = [
    "frame_idx",
    "timestamp_s",
    "track_id",
    "station_id",
    "l_wrist_x",
    "l_wrist_y",
    "l_wrist_vis",
    "r_wrist_x",
    "r_wrist_y",
    "r_wrist_vis",
    "l_elbow_x",
    "l_elbow_y",
    "l_elbow_vis",
    "r_elbow_x",
    "r_elbow_y",
    "r_elbow_vis",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def fmt_coord(val: float | None) -> str:
    """Blank string for null — never '0.0' for missing."""
    if val is None:
        return ""
    return f"{val:.4f}"


def landmark_xy_vis(
    landmarks,
    idx: int,
    crop_x1: int,
    crop_y1: int,
    crop_w: int,
    crop_h: int,
    vis_thresh: float,
) -> tuple[float | None, float | None, float]:
    """Return full-frame pixel coords + visibility; null xy if below threshold."""
    lm = landmarks.landmark[idx]
    vis = float(lm.visibility)
    if vis < vis_thresh:
        return None, None, vis
    x = crop_x1 + float(lm.x) * crop_w
    y = crop_y1 + float(lm.y) * crop_h
    return x, y, vis


def write_checkpoint(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pose extraction (MediaPipe)")
    parser.add_argument("--run-name", default="fasttracker_full")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None, help="Debug: stop after N video frames")
    args = parser.parse_args()

    cfg = load_yaml(ROOT / "config.yaml")
    station_map = load_yaml(ROOT / cfg["pose"]["station_map"])
    pose_cfg = cfg["pose"]

    video_path = ROOT / cfg["video"]["path"]
    tracks_csv = ROOT / pose_cfg["tracks_csv"]
    out_dir = ROOT / cfg["paths"]["pose"] / args.run_name
    ckpt_dir = ROOT / cfg["paths"]["checkpoints"] / f"pose_{args.run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "pose_landmarks.csv"
    ckpt_path = ckpt_dir / "checkpoint.json"

    stride = int(pose_cfg["frame_stride"])
    vis_thresh = float(pose_cfg["visibility_threshold"])
    pad = int(pose_cfg["crop_pad_px"])
    real_ids = set(int(x) for x in pose_cfg["real_track_ids"])
    track_to_station = {int(k): int(v) for k, v in station_map["track_to_station"].items()}
    ckpt_every = int(pose_cfg.get("checkpoint_every", 500))

    print("=" * 60)
    print("Pose extraction — MediaPipe")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Tracks: {tracks_csv}")
    print(f"Output: {out_csv}")
    print(f"frame_stride={stride}  vis_thresh={vis_thresh}  real_ids={sorted(real_ids)}")
    print("Null policy: vis < thresh -> blank x/y (not zero). Movement must use UNKNOWN if sparse.")

    tracks = pd.read_csv(tracks_csv)
    tracks = tracks[tracks["track_id"].isin(real_ids)].copy()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: cannot open {video_path}")
        return 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or float(cfg["video"]["fps"])
    end_frame = total_frames if args.max_frames is None else min(total_frames, args.max_frames)

    start_frame = 0
    if args.fresh:
        if out_csv.exists():
            out_csv.unlink()
        if ckpt_path.exists():
            ckpt_path.unlink()
    elif ckpt_path.exists():
        with open(ckpt_path, encoding="utf-8") as f:
            ck = json.load(f)
        start_frame = int(ck["next_frame"])
        print(f"Resuming from frame {start_frame}")

    if not out_csv.exists():
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADER)

    pose = mp.solutions.pose.Pose(
        static_image_mode=True,  # per-crop; avoid cross-person tracking bleed
        model_complexity=int(pose_cfg.get("model_complexity", 1)),
        enable_segmentation=False,
        min_detection_confidence=0.5,
    )

    # Align start to stride grid
    if start_frame % stride != 0:
        start_frame += stride - (start_frame % stride)

    frame_idx = start_frame
    sampled_done = 0
    rows_written = 0
    t0 = time.perf_counter()

    print("Indexing tracks by frame...")
    by_frame: dict[int, pd.DataFrame] = {
        int(fid): g for fid, g in tracks.groupby("frame_idx")
    }

    out_f = open(out_csv, "a", newline="", encoding="utf-8", buffering=1)
    writer = csv.writer(out_f)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    try:
        while frame_idx < end_frame:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % stride == 0:
                h, w = frame.shape[:2]
                ts = frame_idx / fps
                frame_tracks = by_frame.get(frame_idx)

                if frame_tracks is not None and len(frame_tracks):
                    for _, t in frame_tracks.iterrows():
                        tid = int(t["track_id"])
                        station_id = track_to_station.get(tid, -1)
                        x1, y1 = int(t["x1"]), int(t["y1"])
                        x2, y2 = int(t["x2"]), int(t["y2"])
                        x1c, y1c = max(0, x1 - pad), max(0, y1 - pad)
                        x2c, y2c = min(w, x2 + pad), min(h, y2 + pad)
                        crop = frame[y1c:y2c, x1c:x2c]
                        if crop.size == 0:
                            continue
                        ch, cw = crop.shape[:2]
                        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        result = pose.process(rgb)

                        vals: dict[str, float | None] = {}
                        if not result.pose_landmarks:
                            for name in LM:
                                vals[f"{name}_x"] = None
                                vals[f"{name}_y"] = None
                                vals[f"{name}_vis"] = 0.0
                        else:
                            for name, idx in LM.items():
                                x, y, vis = landmark_xy_vis(
                                    result.pose_landmarks,
                                    idx,
                                    x1c,
                                    y1c,
                                    cw,
                                    ch,
                                    vis_thresh,
                                )
                                vals[f"{name}_x"] = x
                                vals[f"{name}_y"] = y
                                vals[f"{name}_vis"] = vis

                        writer.writerow(
                            [
                                frame_idx,
                                f"{ts:.4f}",
                                tid,
                                station_id,
                                fmt_coord(vals["l_wrist_x"]),
                                fmt_coord(vals["l_wrist_y"]),
                                f"{vals['l_wrist_vis']:.4f}",
                                fmt_coord(vals["r_wrist_x"]),
                                fmt_coord(vals["r_wrist_y"]),
                                f"{vals['r_wrist_vis']:.4f}",
                                fmt_coord(vals["l_elbow_x"]),
                                fmt_coord(vals["l_elbow_y"]),
                                f"{vals['l_elbow_vis']:.4f}",
                                fmt_coord(vals["r_elbow_x"]),
                                fmt_coord(vals["r_elbow_y"]),
                                f"{vals['r_elbow_vis']:.4f}",
                            ]
                        )
                        rows_written += 1

                sampled_done += 1

                if sampled_done % ckpt_every == 0:
                    out_f.flush()
                    elapsed = time.perf_counter() - t0
                    rate = sampled_done / elapsed if elapsed > 0 else 0
                    remaining_samples = max(0, (end_frame - frame_idx - 1) // stride)
                    eta_min = (remaining_samples / rate / 60) if rate > 0 else 0
                    write_checkpoint(
                        ckpt_path,
                        {
                            "run_name": args.run_name,
                            "last_frame": frame_idx,
                            "next_frame": frame_idx + 1,
                            "end_frame": end_frame,
                            "sampled_done": sampled_done,
                            "rows_written": rows_written,
                            "frame_stride": stride,
                            "visibility_threshold": vis_thresh,
                            "fps_effective_samples": round(rate, 3),
                            "updated_at": utc_now(),
                        },
                    )
                    print(
                        f"  ckpt frame={frame_idx}/{end_frame}  samples={sampled_done}  "
                        f"rows={rows_written}  {rate:.2f} samp/s  ETA~{eta_min:.1f} min"
                    )

            frame_idx += 1

    except KeyboardInterrupt:
        out_f.flush()
        # Resume from next unread frame; align to stride on restart
        write_checkpoint(
            ckpt_path,
            {
                "run_name": args.run_name,
                "next_frame": frame_idx,
                "interrupted": True,
                "sampled_done": sampled_done,
                "rows_written": rows_written,
                "updated_at": utc_now(),
            },
        )
        print(f"\nInterrupted at frame {frame_idx}. Re-run without --fresh to resume.")
        out_f.close()
        cap.release()
        pose.close()
        return 130
    finally:
        if not out_f.closed:
            out_f.flush()
            out_f.close()
        cap.release()
        pose.close()
    write_checkpoint(
        ckpt_path,
        {
            "run_name": args.run_name,
            "last_frame": end_frame - 1,
            "next_frame": end_frame,
            "end_frame": end_frame,
            "sampled_done": sampled_done,
            "rows_written": rows_written,
            "frame_stride": stride,
            "visibility_threshold": vis_thresh,
            "completed": True,
            "updated_at": utc_now(),
        },
    )
    elapsed = time.perf_counter() - t0
    print(f"\nDone. samples={sampled_done} rows={rows_written} in {elapsed / 60:.1f} min")
    print(f"CSV: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
