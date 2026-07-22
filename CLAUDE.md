# CLAUDE.md — Worker Monitoring PoC

Context for AI assistants (Cursor / Claude) working in this repo.
Read this before changing pipeline code, thresholds, or re-running long jobs.

---

## Goal

Local CPU PoC for **fixed CCTV** of **4 seated tobacco-sorting workers**:

**YOLOv11 detect → track → MediaPipe pose → movement / activity**

Later (out of scope here): company-server face-rec for identity across reappearances; full dashboard product.

Primary video: `20260602_214750.mp4` (gitignored)  
1280×720, ~30 fps, **18,715 frames** (~10.4 min).

---

## Working style (user-mandated)

1. **One step at a time** — verify outputs before the next stage.
2. **Ask clarifying questions** before large/full-video runs.
3. **Test hard cases first** — occlusion / pole region, frames ~**800–2200**.
4. **No silent low-confidence coords** feeding movement as zero motion.
5. **No track stitching / KMeans merge** in this PoC — raw `track_id`s; identity handoff is server-side.
6. Prefer editing existing scripts over inventing parallel pipelines.

---

## Environment

| Item | Value |
|------|--------|
| OS | Windows 10 |
| Hardware | AMD Ryzen 7 5700G, **no NVIDIA GPU** → **CPU only** |
| Python | 3.9 venv: `.venv\` |
| Activate | `.\.venv\Scripts\Activate.ps1` |
| MediaPipe | pin `mediapipe>=0.10.9,<0.10.21` (`mp.solutions` API) |
| Detector | **yolo11s.pt**, conf **0.25**, imgsz 640, class person only |

Shared tunables: `config.yaml`  
Station ↔ track map: `station_map.yaml`

---

## Pipeline stages & scripts

| Step | Script | Role |
|------|--------|------|
| 0 | `scripts/verify_env.py` | Env check |
| 1 | `scripts/01_detect_sanity.py` | Detection sanity / hard-frame scan |
| 2 | `scripts/02_detect_track_full.py` | Checkpointed detect+track (`--tracker`, `--run-name`) |
| 3 | `scripts/03_diagnose_tracks.py` | Fragmentation / Type-B diagnostics |
| 4 | `scripts/04_label_sessions.py` | `real_session` vs `noise` (no merge) |
| 5 | `scripts/05_extract_pose.py` | MediaPipe pose, stride=2, null low-vis wrists |
| 6 | `scripts/06_activity_classify.py` | Scale-normalized rolling movement |
| 7 | `scripts/07_idle_hunt_high_vf.py` | High-VF low-movement idle candidates |
| 8 | `scripts/08_activity_finalize.py` | Stage D finalize + GT-calibrated IDLE |
| 9 | `scripts/09_export_compare_metrics.py` | Cross-folder compare pack |

Trackers: `trackers/bytetrack.yaml`, `trackers/fasttrack.yaml`

---

## Canonical run used for Stage D

**Run name:** `fasttracker_full`

| Artifact | Path |
|----------|------|
| Tracks | `outputs/tracks/fasttracker_full/tracks.csv` |
| Session labels | `outputs/diagnostics/fasttracker_full/track_labels.csv` |
| Pose | `outputs/pose/fasttracker_full/pose_landmarks.csv` (~37,020 rows) |
| Activity | `outputs/activity/fasttracker_full/` |
| Stage D report | `outputs/activity/fasttracker_full/stage_d_report/` |
| Compare pack | `outputs/activity/fasttracker_full/compare_pack/` |

`outputs/` is gitignored (large). Scripts/configs are in git.

---

## Station map (FastTracker — locked)

| Station | Role | track_id(s) | Notes |
|---------|------|-------------|--------|
| 1 | Left (fg-left) | **2** | Continuous full video |
| 2 | Mid / pole | **3** then **16** | Same person; **real absence** ~1151–1927; do **not** merge |
| 3 | Foreground-right | **1** | Near camera, sorting bin |
| 4 | Background-right | **4** | Often occluded by station 3 |

**When reviewing idle strips:**

- `t1_*` → watch **fg-right** (track 1), not left.
- `t2_*` → watch **leftmost foreground** (track 2), **not** the grey-shirt person further back (that is mid / track 16).

Guide: `outputs/sanity/idle_candidates_t1_t2/WHO_IS_TRACK2.md`

---

## Locked technical policies

### Tracking
- Prefer FastTracker full outputs for pose/activity (ByteTrack also run; mid gap physics same).
- **No stitching.** Mid stays IDs 3 and 16.
- Noise IDs (short phantoms) excluded from pose via `station_map.yaml` / labels.

### Pose
- `frame_stride: 2`
- Visibility threshold **0.5**
- If vis &lt; threshold → store **null** x/y, keep `*_vis`
- Never treat null / missing wrist as (0,0) or zero displacement

### Movement / activity
- Scale: **bbox diagonal** normalized wrist displacement
- Window: ~**1.0 s**
- If `valid_fraction < 0.5` → label **UNKNOWN** (not IDLE)
- **ACTIVE / IDLE only** where visual idle GT exists
- Without idle GT → **PENDING_GT** (score kept) or UNKNOWN

### Idle GT definitions
| Decision | Meaning |
|----------|---------|
| **CONFIRM** | Off-task stillness (e.g. phone) — usable for threshold |
| **REJECT** | Working (sorting, inspecting, **pick-and-drop / fluff**) |
| **PARTIAL** | Ambiguous — **exclude** from threshold fit |

Pick-and-drop of product = **working**, not idle.

---

## Idle GT status (this video)

### Batch A — `outputs/sanity/idle_candidates/`
| Cand | Track | Decision |
|------|-------|----------|
| 01, 02 | 4 | REJECT (occlusion ≠ idle) |
| 03, 05 | 1 | REJECT (sorting) |
| **04** | **16** | **CONFIRM** (phone) frames **8014–8712** |
| 06 | 16 | PARTIAL — exclude from fit |

### Batch B — `outputs/sanity/idle_candidates_t1_t2/`
| Cand | Decision |
|------|----------|
| t1_low01–05 | ALL **REJECT** |
| t2_low01–03 | **REJECT** |
| t2_low04–05 | **PARTIAL** (pick/drop; not idle GT) |

**Implication:** No IDLE threshold for tracks 1, 2, 4. Only track **16** is calibrated.

Details: `outputs/sanity/idle_candidates_t1_t2/GT_DECISIONS_T1_T2.md`  
and `outputs/sanity/idle_candidates/GT_DECISIONS.md`

---

## Stage D results (summary)

**Track 16 calibration:** idle p95 × 1.05 → threshold **≈ 0.229**  
- Full track 16: ~**70% ACTIVE / 29% IDLE / 0.5% UNKNOWN**  
- cand04 validation: ~**95% IDLE**

| Track | Labels |
|-------|--------|
| 1, 2, 3, 4 | PENDING_GT + UNKNOWN only |
| 16 | ACTIVE / IDLE / UNKNOWN |

Mid absence 1151–1927: **0** pose/activity rows for tracks 3+16 (real vacancy).

---

## Identity (out of scope in this folder)

Reappearance / cross-session identity (e.g. 3 ↔ 16) is resolved on the **company server**:

- **ElasticFaceArcAug** + FAR-calibrated galleries  
- **Not** the old Milvus pipeline  

This repo only emits track_id + frame/timestamp artifacts for handoff.

A separate sibling pipeline (face-rec first + ByteTrack + ArcFace Re-ID) is better at **who-is-who**.  
This PoC is stronger on **GT-calibrated activity honesty** (avoid false idle from fine sorting).

**Preferred hybrid:** their Re-ID → Person 1–4, then this repo’s movement + idle GT rules.

---

## Compare with another folder

Export:

```powershell
.\.venv\Scripts\python.exe scripts\09_export_compare_metrics.py
```

Compare these files across folders:

- `compare_pack/per_track_movement_percentiles.csv` — best apples-to-apples  
- `compare_pack/gt_window_scores.csv` — same GT windows  
- `compare_pack/COMPARE_METRICS.json`  

Do **not** treat IDLE% as comparable on persons/tracks without shared CONFIRM idle GT.

---

## Hard cases & known findings

1. **Mid gap ~1124–1966:** real seat empty / no mid-band detections — not only tracker failure. Proof frames under `outputs/sanity/gap_raw/`.
2. **Track 4:** high UNKNOWN (~30%) from workbench / occlusion and wrist visibility swings.
3. Histograms of movement are **mostly not bimodal** — do not invent global median ACTIVE/IDLE splits without GT.
4. High-VF “low movement” on tracks 1–2 was still **on-task** in visual review.

---

## What NOT to do

- Do not merge track 3 and 16 locally.
- Do not set IDLE thresholds from REJECT/PARTIAL windows.
- Do not treat occlusion / null wrists as idle.
- Do not commit secrets, tokens, `.env`, video (`*.mp4`), galleries (`Person*.zip`), weights (`*.pt`), or `outputs/`.
- Do not run full-video jobs without user confirmation.
- Do not push to GitHub unless the user explicitly asks.

---

## Git remote

- Repo: https://github.com/anshul-art/WorkerMonitoring-POC  
- Branch: `main`  
- Tracked: scripts, yaml configs, README, requirements, this file  
- Ignored: `.venv/`, `outputs/`, `*.pt`, `*.mp4`, `Person*.zip`

---

## Quick resume checklist

1. Activate `.venv`
2. Confirm video path in `config.yaml`
3. Prefer artifacts under `*/fasttracker_full/`
4. Idle GT locked as above — re-run `08_activity_finalize.py` only if new **CONFIRM** idle windows are added
5. For cross-pipeline compare, refresh `09_export_compare_metrics.py` and diff `compare_pack/`
