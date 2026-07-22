"""Step 3: detection sanity check — find hard frame, compare models, tune conf."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DetSummary:
    frame_idx: int
    model: str
    conf_thresh: float
    count: int
    mean_conf: float
    min_conf: float
    boxes: list[tuple[float, float, float, float, float]]


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_frame(video_path: Path, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def summarize(result, frame_idx: int, model_name: str, conf_thresh: float) -> DetSummary:
    boxes_out: list[tuple[float, float, float, float, float]] = []
    if result.boxes is not None and len(result.boxes):
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), conf in zip(xyxy, confs):
            boxes_out.append((float(x1), float(y1), float(x2), float(y2), float(conf)))
    confs_only = [b[4] for b in boxes_out]
    return DetSummary(
        frame_idx=frame_idx,
        model=model_name,
        conf_thresh=conf_thresh,
        count=len(boxes_out),
        mean_conf=float(np.mean(confs_only)) if confs_only else 0.0,
        min_conf=float(np.min(confs_only)) if confs_only else 0.0,
        boxes=boxes_out,
    )


def predict(model: YOLO, frame: np.ndarray, cfg: dict, conf: float):
    return model.predict(
        frame,
        conf=conf,
        iou=cfg["detection"]["iou"],
        classes=cfg["model"]["classes"],
        imgsz=cfg["model"]["imgsz"],
        device=cfg["model"]["device"],
        verbose=False,
    )[0]


def draw_and_save(result, out_path: Path, title: str) -> None:
    annotated = result.plot()
    cv2.putText(
        annotated,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)


def find_hard_frame(model: YOLO, video_path: Path, cfg: dict, stride: int) -> tuple[int, list[DetSummary]]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    conf = cfg["detection"]["conf"]
    scan_rows: list[DetSummary] = []

    print(f"\nScanning for hard frame (yolo11s, conf={conf}, every {stride} frames)...")
    t0 = time.perf_counter()
    frame_idx = 0
    while frame_idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        result = predict(model, frame, cfg, conf)
        scan_rows.append(summarize(result, frame_idx, "yolo11s-scan", conf))
        frame_idx += stride
    cap.release()

    elapsed = time.perf_counter() - t0
    print(f"  Scanned {len(scan_rows)} sample frames in {elapsed:.1f}s")

    # Hardest = fewest detections; tie-break by lowest mean confidence
    hard = min(scan_rows, key=lambda r: (r.count, r.mean_conf))
    print(
        f"  Hardest sample: frame {hard.frame_idx} "
        f"({hard.count} persons, mean conf={hard.mean_conf:.3f})"
    )
    return hard.frame_idx, scan_rows


def print_boxes(summary: DetSummary) -> None:
    print(
        f"  frame={summary.frame_idx} model={summary.model} conf>={summary.conf_thresh:.2f} "
        f"-> {summary.count} person(s)"
    )
    for i, (x1, y1, x2, y2, conf) in enumerate(summary.boxes, 1):
        w, h = x2 - x1, y2 - y1
        print(f"    #{i}: conf={conf:.3f}  box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})  size={w:.0f}x{h:.0f}")


def recommend_conf(hard_summaries: list[DetSummary], expected: int) -> tuple[float, str]:
    """Pick lowest conf that still hits expected count on hard frame with yolo11s."""
    s_rows = [s for s in hard_summaries if s.model == "yolo11s"]
    s_rows.sort(key=lambda s: s.conf_thresh)

    for s in s_rows:
        if s.count >= expected:
            return s.conf_thresh, (
                f"conf={s.conf_thresh:.2f} finds {s.count}/{expected} on the hardest sampled frame."
            )

    best = max(s_rows, key=lambda s: s.count)
    return best.conf_thresh, (
        f"Even at conf={best.conf_thresh:.2f}, only {best.count}/{expected} on hard frame — "
        "may need lower conf or accept detection gaps (ByteTrack track_buffer helps)."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Detection sanity check (Step 3)")
    parser.add_argument("--skip-scan", action="store_true", help="Reuse hard frame from last scan output")
    args = parser.parse_args()

    cfg = load_config()
    expected = cfg["video"]["expected_workers"]
    video_path = ROOT / cfg["video"]["path"]
    out_dir = ROOT / cfg["paths"]["sanity"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 3 — Detection sanity check")
    print("=" * 60)
    print(f"Expected workers: {expected}")
    print(f"Video: {video_path.name}")

    model_s = YOLO(cfg["model"]["weights"])
    model_n = YOLO(cfg["model"]["sanity_compare_weights"])

    easy_idx = cfg["sanity"]["easy_frame"]
    stride = cfg["sanity"]["sample_stride"]

    if args.skip_scan:
        hard_idx = easy_idx
        print("\n--skip-scan: using easy frame only (hard-frame scan skipped)")
    else:
        hard_idx, scan_rows = find_hard_frame(model_s, video_path, cfg, stride)
        scan_csv = out_dir / "scan_samples.csv"
        with open(scan_csv, "w", encoding="utf-8") as f:
            f.write("frame_idx,count,mean_conf,min_conf\n")
            for r in scan_rows:
                f.write(f"{r.frame_idx},{r.count},{r.mean_conf:.4f},{r.min_conf:.4f}\n")
        print(f"  Scan log: {scan_csv}")

    frames = {easy_idx: "easy", hard_idx: "hard"}
    if easy_idx == hard_idx:
        frames = {easy_idx: "easy"}

    base_conf = cfg["detection"]["conf"]
    compare_rows: list[DetSummary] = []

    print(f"\n--- Model comparison at conf={base_conf} ---")
    for idx, label in frames.items():
        frame = read_frame(video_path, idx)
        if frame is None:
            print(f"ERROR: could not read frame {idx}")
            return 1
        timestamp_s = idx / cfg["video"]["fps"]
        print(f"\n[{label.upper()}] frame {idx} (~{timestamp_s:.1f}s)")

        for model, name in ((model_s, "yolo11s"), (model_n, "yolo11n")):
            t0 = time.perf_counter()
            result = predict(model, frame, cfg, base_conf)
            ms = (time.perf_counter() - t0) * 1000
            summary = summarize(result, idx, name, base_conf)
            compare_rows.append(summary)
            print_boxes(summary)
            print(f"  inference: {ms:.0f} ms")
            img_name = f"{label}_f{idx}_{name}_conf{base_conf:.2f}.jpg"
            draw_and_save(result, out_dir / img_name, f"{name} conf>={base_conf:.2f} | {summary.count} persons")

    hard_conf_sweep: list[DetSummary] = []
    if hard_idx != easy_idx:
        print(f"\n--- Confidence sweep on HARD frame {hard_idx} (yolo11s only) ---")
        print("  Tradeoff: lower conf -> more detections but more false positives/noise for ByteTrack")
        hard_frame = read_frame(video_path, hard_idx)
        for conf in cfg["sanity"]["conf_sweep"]:
            result = predict(model_s, hard_frame, cfg, conf)
            summary = summarize(result, hard_idx, "yolo11s", conf)
            hard_conf_sweep.append(summary)
            print_boxes(summary)
            draw_and_save(
                result,
                out_dir / f"hard_f{hard_idx}_yolo11s_conf{conf:.2f}.jpg",
                f"yolo11s conf>={conf:.2f} | {summary.count} persons",
            )

    # Decision summary
    print("\n" + "=" * 60)
    print("DECISION SUMMARY")
    print("=" * 60)

    easy_s = next(r for r in compare_rows if r.model == "yolo11s" and r.frame_idx == easy_idx)
    easy_n = next(r for r in compare_rows if r.model == "yolo11n" and r.frame_idx == easy_idx)
    print(f"\n1) Model choice (easy frame {easy_idx}):")
    print(f"   yolo11s: {easy_s.count} persons, min conf={easy_s.min_conf:.3f}")
    print(f"   yolo11n: {easy_n.count} persons, min conf={easy_n.min_conf:.3f}")
    if easy_n.count < easy_s.count:
        print("   -> yolo11s finds more boxes; nano is weaker even on easy frame.")
    elif easy_n.count == easy_s.count and easy_n.min_conf < easy_s.min_conf - 0.05:
        print("   -> Same count, but nano confidences are lower (less stable for tracking).")
    else:
        print("   -> Both OK on easy frame; hard frame comparison matters more.")

    if hard_idx != easy_idx:
        hard_s = next(r for r in compare_rows if r.model == "yolo11s" and r.frame_idx == hard_idx)
        hard_n = next(r for r in compare_rows if r.model == "yolo11n" and r.frame_idx == hard_idx)
        print(f"\n2) Model choice (hard frame {hard_idx}):")
        print(f"   yolo11s: {hard_s.count} persons, min conf={hard_s.min_conf:.3f}")
        print(f"   yolo11n: {hard_n.count} persons, min conf={hard_n.min_conf:.3f}")
        if hard_s.count > hard_n.count:
            print("   -> yolo11s wins on occlusion — fewer detection gaps for ByteTrack.")
        elif hard_s.count == hard_n.count:
            print("   -> Same count; prefer yolo11s if its min confidence is higher (more stable tracks).")
        else:
            print("   -> Unexpected: review annotated images in outputs/sanity/")

        rec_conf, reason = recommend_conf(hard_conf_sweep, expected)
        print(f"\n3) Confidence threshold (from hard-frame sweep):")
        print(f"   Recommended starting conf: {rec_conf:.2f}")
        print(f"   Why: {reason}")
        print("   How this helps ByteTrack:")
        print("   - Missed detections -> track goes 'lost' -> new ID after gap (fragmentation)")
        print("   - Too-low conf -> spurious boxes -> short noisy tracks")
        print("   - Goal: conf low enough to keep 4 boxes on hard frames, high enough to drop noise")

    print(f"\nAnnotated images saved to: {out_dir}")
    print("Review the JPGs, then confirm before Step 4 (full-video detect+track).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
