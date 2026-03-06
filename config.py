"""
Central configuration for the football analysis pipeline.

Adjust these values to match your video source and pitch dimensions
before running main.py.
"""

import cv2

# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------
FRAME_RATE: int = 24  # frames per second of the input video

# ---------------------------------------------------------------------------
# YOLO / tracker
# ---------------------------------------------------------------------------
DETECTION_CONF_THRESHOLD: float = 0.1
DETECTION_BATCH_SIZE: int = 20

# ---------------------------------------------------------------------------
# Camera movement estimator (Lucas-Kanade optical flow)
# ---------------------------------------------------------------------------
CAMERA_MIN_DISTANCE: int = 5  # pixels — ignore tiny jitter below this

LK_PARAMS: dict = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

# goodFeaturesToTrack parameters
FEATURE_PARAMS: dict = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=3,
    blockSize=7,
)

# Fraction of frame width used as the feature-detection mask strip on each side.
# The mask keeps only the left and right edge bands so that features are stable
# pitch-line references rather than moving players.
CAMERA_MASK_LEFT_FRAC: float = 0.01   # left strip  (≈20 px on 1920-wide video)
CAMERA_MASK_RIGHT_FRAC: float = 0.47  # right strip (≈900–1050 px on 1920-wide video)

# ---------------------------------------------------------------------------
# Player-ball assignment
# ---------------------------------------------------------------------------
MAX_PLAYER_BALL_DISTANCE: int = 70  # pixels

# ---------------------------------------------------------------------------
# Speed and distance estimator
# ---------------------------------------------------------------------------
SPEED_FRAME_WINDOW: int = 5  # number of frames between measurements

# ---------------------------------------------------------------------------
# View transformer — real-world pitch dimensions (metres)
# Defaults correspond to the half-pitch width and depth visible in the
# sample video from the tutorial.
# ---------------------------------------------------------------------------
PITCH_WIDTH_M: float = 68.0
PITCH_LENGTH_M: float = 23.32

# Pixel coordinates of the four pitch corners visible in the camera frame.
# Update these for your own footage.
PITCH_PIXEL_VERTICES: list[list[int]] = [
    [110, 1035],
    [265, 275],
    [910, 260],
    [1640, 915],
]

# ---------------------------------------------------------------------------
# Annotation colours  (BGR)
# ---------------------------------------------------------------------------
COLOUR_REFEREE: tuple[int, int, int] = (0, 255, 255)   # cyan
COLOUR_BALL: tuple[int, int, int] = (0, 255, 0)         # green
COLOUR_BALL_POSSESSION: tuple[int, int, int] = (0, 0, 255)  # red

BALL_CONTROL_OVERLAY_ALPHA: float = 0.4
