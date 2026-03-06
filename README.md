# Football Analysis

A computer-vision pipeline that detects and tracks players, referees, and the ball in football match footage. Built following the [CodeInAJiffy tutorial](https://www.youtube.com/watch?v=neBZ6huolkg) and extended with several improvements.

## What it does

| Step | Component | Description |
|------|-----------|-------------|
| 1 | **Tracker** | YOLO + ByteTrack to detect and track players, referees, and the ball across frames |
| 2 | **Camera movement** | Lucas-Kanade optical flow to compensate for camera pan/tilt |
| 3 | **View transformer** | Homography to map pixel positions to real-world pitch metres |
| 4 | **Team assigner** | K-means colour clustering on jersey crops to separate teams |
| 5 | **Ball assigner** | Nearest-player logic to determine ball possession each frame |
| 6 | **Speed & distance** | Per-player speed (km/h) and cumulative distance (m) from transformed positions |

The annotated output video shows player IDs, team colours, ball possession, camera movement, and speed/distance overlays.

## Improvements over the tutorial

- **`config.py`** — all magic numbers (fps, distances, pitch dimensions, pixel vertices, colours) centralised in one place
- **Frame-relative annotation box** — ball-control HUD is positioned as a fraction of frame size rather than hardcoded `1350, 850` pixels, so it works with any resolution
- **Dynamic camera mask** — feature-detection strips are calculated from frame width fractions instead of fixed pixel columns
- **Typos fixed** — `draw_traingle` → `_draw_triangle`, `ouput` → `output`, `miniumum` → `minimum`, etc.
- **Removed hardcoded player ID hack** — the `if player_id == 91: team_id = 1` workaround is gone; team assignment is purely data-driven
- **`read_video` returns actual FPS** — downstream fps is read from the video file rather than assumed to be 24
- **Type hints** throughout
- **`logging`** instead of bare prints
- **Proper `__init__.py`** for every package — clean imports with no `sys.path.append` hacks

## Setup

```bash
pip install -r requirements.txt
```

Download the pre-trained YOLO model (trained on the football dataset) and sample video from the [Google Drive link in the original repo](https://github.com/abdullahtarek/football_analysis) and place them at:

```
models/best.pt
input_videos/08fd33_4.mp4
```

## Usage

```bash
python main.py
```

The annotated video is written to `output_videos/output.avi`.

On subsequent runs, cached detections and camera-movement stubs in `stubs/` are reused automatically.  Delete the `.pkl` files there to force a fresh run.

## Configuration

Edit `config.py` to change:

- `FRAME_RATE` — override if the stub was built at a different fps than your clip
- `PITCH_PIXEL_VERTICES` — update to the four pitch-corner pixel coordinates for your camera angle
- `PITCH_WIDTH_M` / `PITCH_LENGTH_M` — real-world dimensions of the visible pitch section
- `MAX_PLAYER_BALL_DISTANCE` — possession-assignment threshold in pixels
- Annotation colours and overlay alpha

## Project structure

```
football-analysis/
├── config.py                        # Central config (fps, colours, dimensions, …)
├── main.py                          # Pipeline entry point
├── yolo_inference.py                # Quick YOLO debug script
├── requirements.txt
├── camera_movement_estimator/
├── player_ball_assigner/
├── speed_and_distance_estimator/
├── team_assigner/
├── trackers/
├── utils/
├── view_transformer/
├── input_videos/                    # Place input .mp4 here (git-ignored)
├── output_videos/                   # Generated output (git-ignored)
├── models/                          # YOLO weights (git-ignored)
└── stubs/                           # Cached pickle stubs (git-ignored)
```
