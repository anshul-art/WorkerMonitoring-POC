"""Step 4: full-video person detection + ByteTrack with checkpoint/resume.

CSV columns (designed for Step 5 position-based diagnosis):
  frame_idx, track_id, x1, y1, x2, y2, conf, cx, cy, w, h

Resume notes:
  - Progress is flushed every checkpoint_every frames.
  - On resume, ByteTrack state is reset (new process) so track IDs may restart.
  - We apply id_offset = max(existing track_id) so CSV IDs stay unique.
  - Resume seams are logged to a sidecar JSONL for Step 5 to flag.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

CSV_HEADER = "frame_idx,track_id,x1,y1,x2,y2,conf,cx,cy,w,h\n"


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs(cfg: dict, run_name: str, tracker_name: str) -> dict[str, Path]:
    tracks = ROOT / cfg["paths"]["tracks"] / run_name
    ckpt = ROOT / cfg["paths"]["checkpoints"] / run_name
    tracks.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    tracker_path = ROOT / "trackers" / tracker_name
    if not tracker_path.exists():
        # fall back to Ultralytics bundled name
        tracker_path = tracker_name
    return {
        "tracks": tracks,
        "ckpt": ckpt,
        "csv": tracks / "tracks.csv",
        "meta": ckpt / "checkpoint.json",
        "resumes": ckpt / "resume_events.jsonl",
        "tracker": tracker_path,
    }


def read_checkpoint(meta_path: Path) -> dict | None:
    if not meta_path.exists():
        return None
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def write_checkpoint(meta_path: Path, payload: dict) -> None:
    tmp = meta_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(meta_path)


def truncate_csv_to_frame(csv_path: Path, last_frame: int) -> int:
    """Keep header + rows with frame_idx <= last_frame. Returns rows kept."""
    if not csv_path.exists():
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(CSV_HEADER)
        return 0

    with open(csv_path, encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(CSV_HEADER)
        return 0

    kept = [lines[0]]
    for line in lines[1:]:
        parts = line.strip().split(",")
        if not parts or not parts[0]:
            continue
        try:
            if int(parts[0]) <= last_frame:
                kept.append(line if line.endswith("\n") else line + "\n")
        except ValueError:
            continue

    tmp = csv_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(kept)
    tmp.replace(csv_path)
    return len(kept) - 1


def max_track_id_in_csv(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    max_id = 0
    with open(csv_path, encoding="utf-8") as f:
        next(f, None)  # header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            try:
                max_id = max(max_id, int(parts[1]))
            except ValueError:
                continue
    return max_id


def append_resume_event(path: Path, event: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def format_rows(result, frame_idx: int, id_offset: int) -> list[str]:
    rows: list[str] = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return rows
    if boxes.id is None:
        # Detections without assigned track IDs — skip (ByteTrack not ready yet)
        return rows

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    ids = boxes.id.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), conf, tid in zip(xyxy, confs, ids):
        tid = int(tid) + id_offset
        w = float(x2 - x1)
        h = float(y2 - y1)
        cx = float(x1 + w / 2.0)
        cy = float(y1 + h / 2.0)
        rows.append(
            f"{frame_idx},{tid},{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f},"
            f"{conf:.4f},{cx:.2f},{cy:.2f},{w:.2f},{h:.2f}\n"
        )
    return rows


def run(
    run_name: str,
    start_frame: int | None,
    max_frames: int | None,
    force_fresh: bool,
    tracker_name: str | None = None,
) -> int:
    cfg = load_config()
    tracker_name = tracker_name or cfg.get("tracking", {}).get("tracker", "bytetrack.yaml")
    # Prefer project-local copy if present
    local = ROOT / "trackers" / Path(tracker_name).name
    if local.exists():
        tracker_name = local.name
    paths = ensure_dirs(cfg, run_name, tracker_name)
    video_path = ROOT / cfg["video"]["path"]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: cannot open video {video_path}")
        return 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or float(cfg["video"]["fps"])
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ckpt = None if force_fresh else read_checkpoint(paths["meta"])
    id_offset = 0
    resume_from = start_frame if start_frame is not None else 0
    end_frame_from_ckpt: int | None = None

    if force_fresh and paths["csv"].exists():
        paths["csv"].unlink()
        if paths["meta"].exists():
            paths["meta"].unlink()
        if paths["resumes"].exists():
            paths["resumes"].unlink()
        ckpt = None

    if ckpt is not None and start_frame is None:
        last_saved = int(ckpt["last_frame"])
        resume_from = last_saved + 1
        # Drop any rows written after the last flush (kill can overrun checkpoint)
        n_kept = truncate_csv_to_frame(paths["csv"], last_saved)
        id_offset = max_track_id_in_csv(paths["csv"])
        if "end_frame_target" in ckpt:
            end_frame_from_ckpt = int(ckpt["end_frame_target"])
        print(
            f"Resuming from checkpoint: next frame={resume_from}, "
            f"id_offset={id_offset}, csv_rows_kept={n_kept}"
        )
        append_resume_event(
            paths["resumes"],
            {
                "ts": utc_now(),
                "resume_from_frame": resume_from,
                "id_offset": id_offset,
                "csv_rows_kept": n_kept,
                "reason": "checkpoint",
            },
        )
    elif paths["csv"].exists() and resume_from > 0 and not force_fresh:
        # Manual start with existing CSV (e.g. after kill without final flush)
        id_offset = max_track_id_in_csv(paths["csv"])
        print(f"Continuing existing CSV: start={resume_from}, id_offset={id_offset}")
        append_resume_event(
            paths["resumes"],
            {
                "ts": utc_now(),
                "resume_from_frame": resume_from,
                "id_offset": id_offset,
                "reason": "manual_continue",
            },
        )

    # Prefer original run target from checkpoint so resume keeps the same end frame
    if end_frame_from_ckpt is not None:
        end_frame = min(total_frames, end_frame_from_ckpt)
    elif max_frames is not None:
        end_frame = min(total_frames, resume_from + max_frames)
    else:
        end_frame = total_frames

    if resume_from >= end_frame:
        print(f"Nothing to do: resume_from={resume_from} >= end_frame={end_frame}")
        cap.release()
        return 0

    if not paths["csv"].exists():
        with open(paths["csv"], "w", encoding="utf-8") as f:
            f.write(CSV_HEADER)

    print("=" * 60)
    print("Step 4 — Detect + ByteTrack (checkpointed)")
    print("=" * 60)
    print(f"Run: {run_name}")
    print(f"Video: {video_path.name} ({width}x{height}, {fps:.2f} fps, {total_frames} frames)")
    print(f"Range: frames [{resume_from}, {end_frame})")
    print(f"Model: {cfg['model']['weights']}  conf={cfg['detection']['conf']}")
    print(f"Tracker: {paths['tracker']}")
    print(f"CSV: {paths['csv']}")
    print(f"Checkpoint every: {cfg['processing']['checkpoint_every']} frames")
    print(f"id_offset: {id_offset}")

    # Use as many CPU threads as available on this Ryzen
    torch.set_num_threads(max(1, torch.get_num_threads()))

    model = YOLO(cfg["model"]["weights"])
    checkpoint_every = int(cfg["processing"]["checkpoint_every"])

    cap.set(cv2.CAP_PROP_POS_FRAMES, resume_from)
    frame_idx = resume_from
    rows_since_ckpt = 0
    max_tid_seen = id_offset
    t0 = time.perf_counter()
    frames_done = 0

    csv_f = open(paths["csv"], "a", encoding="utf-8", buffering=1)

    try:
        while frame_idx < end_frame:
            ok, frame = cap.read()
            if not ok:
                print(f"\nVideo read ended early at frame {frame_idx}")
                break

            result = model.track(
                frame,
                persist=True,
                conf=cfg["detection"]["conf"],
                iou=cfg["detection"]["iou"],
                classes=cfg["model"]["classes"],
                imgsz=cfg["model"]["imgsz"],
                device=cfg["model"]["device"],
                tracker=str(paths["tracker"]),
                verbose=False,
            )[0]

            rows = format_rows(result, frame_idx, id_offset)
            for row in rows:
                csv_f.write(row)
                tid = int(row.split(",")[1])
                if tid > max_tid_seen:
                    max_tid_seen = tid
            rows_since_ckpt += len(rows)
            frames_done += 1

            if frames_done % checkpoint_every == 0 or frame_idx == end_frame - 1:
                csv_f.flush()
                elapsed = time.perf_counter() - t0
                fps_eff = frames_done / elapsed if elapsed > 0 else 0
                remaining = end_frame - frame_idx - 1
                eta_min = (remaining / fps_eff / 60) if fps_eff > 0 else 0
                write_checkpoint(
                    paths["meta"],
                    {
                        "run_name": run_name,
                        "video": str(video_path.name),
                        "last_frame": frame_idx,
                        "next_frame": frame_idx + 1,
                        "end_frame_target": end_frame,
                        "total_frames": total_frames,
                        "max_track_id": max_tid_seen,
                        "id_offset": id_offset,
                        "frames_done_this_session": frames_done,
                        "fps_effective": round(fps_eff, 3),
                        "updated_at": utc_now(),
                        "config": {
                            "weights": cfg["model"]["weights"],
                            "conf": cfg["detection"]["conf"],
                            "iou": cfg["detection"]["iou"],
                            "imgsz": cfg["model"]["imgsz"],
                        },
                    },
                )
                print(
                    f"  ckpt frame={frame_idx}/{end_frame - 1}  "
                    f"session_frames={frames_done}  "
                    f"{fps_eff:.2f} fps  ETA~{eta_min:.1f} min  "
                    f"max_tid={max_tid_seen}"
                )

            frame_idx += 1
    except KeyboardInterrupt:
        csv_f.flush()
        write_checkpoint(
            paths["meta"],
            {
                "run_name": run_name,
                "video": str(video_path.name),
                "last_frame": frame_idx - 1 if frames_done else resume_from - 1,
                "next_frame": frame_idx,
                "end_frame_target": end_frame,
                "total_frames": total_frames,
                "max_track_id": max_tid_seen,
                "id_offset": id_offset,
                "frames_done_this_session": frames_done,
                "interrupted": True,
                "updated_at": utc_now(),
            },
        )
        print(f"\nInterrupted at frame {frame_idx}. Checkpoint saved — re-run to resume.")
        csv_f.close()
        cap.release()
        return 130
    finally:
        if not csv_f.closed:
            csv_f.flush()
            csv_f.close()
        cap.release()

    elapsed = time.perf_counter() - t0
    print(f"\nDone. Frames this session: {frames_done} in {elapsed / 60:.1f} min")
    print(f"CSV: {paths['csv']}")
    print(f"Checkpoint: {paths['meta']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 4: detect + track with checkpoints")
    parser.add_argument("--run-name", default="full", help="Output subfolder name under outputs/tracks/")
    parser.add_argument("--start-frame", type=int, default=None, help="Override start frame")
    parser.add_argument("--max-frames", type=int, default=None, help="Process at most N frames from start")
    parser.add_argument("--fresh", action="store_true", help="Ignore checkpoint and overwrite CSV")
    parser.add_argument(
        "--tracker",
        default=None,
        help="Tracker yaml name (e.g. bytetrack.yaml, fasttrack.yaml). Default from config.yaml",
    )
    args = parser.parse_args()
    return run(args.run_name, args.start_frame, args.max_frames, args.fresh, args.tracker)


if __name__ == "__main__":
    raise SystemExit(main())
