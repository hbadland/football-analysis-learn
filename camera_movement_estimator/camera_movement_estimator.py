import logging
import os
import pickle
from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils import measure_distance, measure_xy_distance
from config import (
    CAMERA_MIN_DISTANCE,
    LK_PARAMS,
    FEATURE_PARAMS,
    CAMERA_MASK_LEFT_FRAC,
    CAMERA_MASK_RIGHT_FRAC,
)

logger = logging.getLogger(__name__)


class CameraMovementEstimator:
    """Estimates per-frame camera translation using Lucas-Kanade optical flow."""

    def __init__(self, frame: np.ndarray) -> None:
        self.minimum_distance = CAMERA_MIN_DISTANCE

        h, w = frame.shape[:2]

        # Build a mask that restricts feature detection to narrow vertical
        # strips near the left and right edges of the frame.  These strips
        # contain stable pitch-line features regardless of player positions.
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        left_px = max(1, int(w * CAMERA_MASK_LEFT_FRAC))
        right_start = int(w * CAMERA_MASK_RIGHT_FRAC)
        right_end = min(w, right_start + int(w * 0.08))
        mask[:, :left_px] = 1
        mask[:, right_start:right_end] = 1

        self._feature_params = dict(**FEATURE_PARAMS, mask=mask)
        self._lk_params = LK_PARAMS

    # ------------------------------------------------------------------

    def get_camera_movement(
        self,
        frames: List[np.ndarray],
        read_from_stub: bool = False,
        stub_path: Optional[str] = None,
    ) -> List[List[float]]:
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        camera_movement: List[List[float]] = [[0.0, 0.0]] * len(frames)

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self._feature_params)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self._lk_params
            )

            max_distance = 0.0
            cam_x, cam_y = 0.0, 0.0

            if new_features is not None and old_features is not None:
                for new, old in zip(new_features, old_features):
                    new_pt = new.ravel()
                    old_pt = old.ravel()
                    dist = measure_distance(new_pt, old_pt)
                    if dist > max_distance:
                        max_distance = dist
                        cam_x, cam_y = measure_xy_distance(old_pt, new_pt)

            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [cam_x, cam_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self._feature_params)

            old_gray = frame_gray.copy()

        if stub_path:
            with open(stub_path, "wb") as f:
                pickle.dump(camera_movement, f)
            logger.info("Saved camera movement stubs to %s", stub_path)

        return camera_movement

    def add_adjusted_positions_to_tracks(
        self, tracks: dict, camera_movement_per_frame: List[List[float]]
    ) -> None:
        """Subtract camera motion from each tracked object's raw position."""
        for object_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                cam_x, cam_y = camera_movement_per_frame[frame_num]
                for track_id, track_info in track.items():
                    x, y = track_info["position"]
                    tracks[object_name][frame_num][track_id]["position_adjusted"] = (
                        x - cam_x,
                        y - cam_y,
                    )

    def draw_camera_movement(
        self,
        frames: List[np.ndarray],
        camera_movement_per_frame: List[List[float]],
    ) -> List[np.ndarray]:
        output_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            x_mov, y_mov = camera_movement_per_frame[frame_num]
            cv2.putText(frame, f"Camera X: {x_mov:.2f}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.putText(frame, f"Camera Y: {y_mov:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            output_frames.append(frame)

        return output_frames
