"""Step 2: verify local environment — CPU PyTorch, Ultralytics, YOLO11s load."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    print("=" * 60)
    print("Worker Monitoring PoC — environment check")
    print("=" * 60)
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")

    import torch

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()} (expect False — CPU-only)")
    print(f"CPU threads: {torch.get_num_threads()}")

    import cv2
    import pandas
    import yaml
    from ultralytics import YOLO

    print(f"OpenCV: {cv2.__version__}")
    print(f"pandas: {pandas.__version__}")
    print(f"Ultralytics: {__import__('ultralytics').__version__}")

    config_path = ROOT / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    weights = cfg["model"]["weights"]
    video_path = ROOT / cfg["video"]["path"]

    print(f"\nConfig: {config_path.name}")
    print(f"Model weights: {weights}")
    print(f"Video: {video_path} — exists={video_path.exists()}")

    if video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"Video props: {w}x{h}, {fps:.2f} fps, {n} frames (~{n / fps / 60:.1f} min)")

    print(f"\nLoading {weights} (first run downloads ~20 MB)...")
    model = YOLO(weights)
    print(f"Model loaded: {model.model.__class__.__name__}, device={cfg['model']['device']}")

    # Single-frame smoke test — no full-video work
    if video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        ok, frame = cap.read()
        cap.release()
        if ok:
            import time

            t0 = time.perf_counter()
            results = model.predict(
                frame,
                conf=cfg["detection"]["conf"],
                iou=cfg["detection"]["iou"],
                classes=cfg["model"]["classes"],
                imgsz=cfg["model"]["imgsz"],
                device=cfg["model"]["device"],
                verbose=False,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            n_persons = len(results[0].boxes) if results[0].boxes is not None else 0
            est_full_min = (n / fps) * (elapsed_ms / 1000) / 60 if video_path.exists() else 0
            print(f"\nFrame-0 smoke test: {n_persons} person(s) detected in {elapsed_ms:.0f} ms")
            print(f"Rough full-video detection estimate: ~{est_full_min:.0f} min (CPU, single pass)")

    print("\nEnvironment OK — ready for detection sanity check (Step 3).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
