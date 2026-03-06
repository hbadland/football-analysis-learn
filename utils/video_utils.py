import logging
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def read_video(video_path: str) -> Tuple[List[np.ndarray], float]:
    """Read all frames from a video file.

    Returns:
        frames: list of BGR frames.
        fps: frames per second of the source video.

    Raises:
        FileNotFoundError: if the video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames: List[np.ndarray] = []

    while True:
        success, frame = cap.read()
        if not success:
            break
        frames.append(frame)

    cap.release()
    logger.info("Read %d frames at %.1f fps from %s", len(frames), fps, video_path)
    return frames, fps


def save_video(output_video_frames: List[np.ndarray], output_video_path: str, fps: float = 24.0) -> None:
    """Write a list of BGR frames to a video file (XVID codec, .avi)."""
    if not output_video_frames:
        raise ValueError("No frames to write.")

    height, width = output_video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame in output_video_frames:
        writer.write(frame)

    writer.release()
    logger.info("Saved %d frames to %s", len(output_video_frames), output_video_path)
