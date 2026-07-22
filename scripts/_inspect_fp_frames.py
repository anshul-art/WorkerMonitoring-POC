"""Map worker slots frame 0 vs missing at 1500."""

from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
model = YOLO(cfg["model"]["weights"])
video = ROOT / cfg["video"]["path"]
cap = cv2.VideoCapture(str(video))


def dets(frame):
    r = model.predict(
        frame, conf=0.25, iou=0.45, classes=[0], imgsz=640, device="cpu", verbose=False
    )[0]
    out = []
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        out.append(((x1 + x2) / 2, (y1 + y2) / 2, float(box.conf)))
    return out


cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
_, f0 = cap.read()
refs = sorted(dets(f0), key=lambda t: t[0])
labels = ["left", "center", "foreground-right", "background-right"]
print("Frame 0 worker slots (left -> right):")
for lab, (cx, cy, c) in zip(labels, refs):
    print(f"  {lab}: center=({cx:.0f},{cy:.0f}) conf={c:.3f}")

cap.set(cv2.CAP_PROP_POS_FRAMES, 1500)
_, f1500 = cap.read()
d1500 = dets(f1500)
print("\nFrame 1500 detections:")
for cx, cy, c in d1500:
    print(f"  center=({cx:.0f},{cy:.0f}) conf={c:.3f}")

print("\nFrame 1500 slot match (nearest det, missing if >120px):")
for lab, (rcx, rcy, _) in zip(labels, refs):
    dists = [(((cx - rcx) ** 2 + (cy - rcy) ** 2) ** 0.5, cx, cy, c) for cx, cy, c in d1500]
    dist, cx, cy, c = min(dists, key=lambda t: t[0])
    status = "DETECTED" if dist < 120 else "MISSING"
    print(f"  {lab}: {status} nearest_dist={dist:.0f}px")

five_frames = []
for idx in range(0, 18715, 500):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    if not ok:
        break
    r = model.predict(
        fr, conf=0.25, iou=0.45, classes=[0], imgsz=640, device="cpu", verbose=False
    )[0]
    n = 0 if r.boxes is None else len(r.boxes)
    if n >= 5:
        five_frames.append(idx)
print("\nSample frames with >=5 detections (stride 500):", five_frames)
cap.release()
