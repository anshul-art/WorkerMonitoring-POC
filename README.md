# WorkerMonitoring-POC

Local PoC for fixed CCTV worker monitoring: **detect → track → pose → movement/activity**.

## Pipeline (this repo)

1. YOLOv11 person detection  
2. Tracking (ByteTrack / FastTracker) — **no track stitching**  
3. MediaPipe pose (wrists), null low-visibility coords  
4. Scale-normalized rolling movement + GT-calibrated ACTIVE/IDLE  
5. Identity reappearance → company server face-rec (out of scope here)

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place the source video in the project root (gitignored). Configure paths in `config.yaml` / `station_map.yaml`.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/verify_env.py` | Environment check |
| `scripts/01_detect_sanity.py` | Detection sanity |
| `scripts/02_detect_track_full.py` | Full detect+track (checkpointed) |
| `scripts/03_diagnose_tracks.py` | Track diagnostics |
| `scripts/04_label_sessions.py` | real_session vs noise |
| `scripts/05_extract_pose.py` | Pose extraction |
| `scripts/06_activity_classify.py` | Movement scores |
| `scripts/07_idle_hunt_high_vf.py` | Idle candidates |
| `scripts/08_activity_finalize.py` | GT-calibrated finalize |
| `scripts/09_export_compare_metrics.py` | Cross-folder compare pack |

## Activity policy

- `valid_fraction < 0.5` → **UNKNOWN** (not IDLE)  
- ACTIVE/IDLE only where visual idle GT exists  
- Sorting / pick-and-drop ≠ idle  

## Compare pack

After Stage D: `python scripts/09_export_compare_metrics.py`  
→ `outputs/activity/<run>/compare_pack/`
