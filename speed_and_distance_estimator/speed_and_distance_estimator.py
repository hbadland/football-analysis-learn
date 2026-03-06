import logging
from typing import List

import cv2
import numpy as np

from utils import measure_distance, get_foot_position
from config import SPEED_FRAME_WINDOW, FRAME_RATE

logger = logging.getLogger(__name__)

# Objects we do not calculate speed for.
_SKIP_OBJECTS = {"ball", "referees"}


class SpeedAndDistanceEstimator:
    """Computes per-player speed (km/h) and cumulative distance (m)."""

    def __init__(self, frame_window: int = SPEED_FRAME_WINDOW, frame_rate: float = FRAME_RATE) -> None:
        self.frame_window = frame_window
        self.frame_rate = frame_rate

    def add_speed_and_distance_to_tracks(self, tracks: dict) -> None:
        total_distance: dict = {}

        for object_name, object_tracks in tracks.items():
            if object_name in _SKIP_OBJECTS:
                continue

            num_frames = len(object_tracks)
            for frame_num in range(0, num_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, num_frames - 1)

                for track_id in object_tracks[frame_num]:
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_pos = object_tracks[frame_num][track_id].get("position_transformed")
                    end_pos = object_tracks[last_frame][track_id].get("position_transformed")

                    if start_pos is None or end_pos is None:
                        continue

                    distance_m = measure_distance(start_pos, end_pos)
                    elapsed_s = (last_frame - frame_num) / self.frame_rate
                    speed_ms = distance_m / elapsed_s if elapsed_s > 0 else 0.0
                    speed_kmh = speed_ms * 3.6

                    total_distance.setdefault(object_name, {})
                    total_distance[object_name].setdefault(track_id, 0.0)
                    total_distance[object_name][track_id] += distance_m

                    for f in range(frame_num, last_frame + 1):
                        if track_id not in object_tracks[f]:
                            continue
                        object_tracks[f][track_id]["speed"] = speed_kmh
                        object_tracks[f][track_id]["distance"] = total_distance[object_name][track_id]

    def draw_speed_and_distance(
        self,
        frames: List[np.ndarray],
        tracks: dict,
    ) -> List[np.ndarray]:
        output_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            for object_name, object_tracks in tracks.items():
                if object_name in _SKIP_OBJECTS:
                    continue
                for track_id, track_info in object_tracks[frame_num].items():
                    speed = track_info.get("speed")
                    distance = track_info.get("distance")
                    if speed is None:
                        continue

                    bbox = track_info.get("bbox")
                    if bbox is None:
                        continue

                    foot_x, foot_y = get_foot_position(bbox)
                    label_pos = (foot_x, foot_y + 40)

                    cv2.putText(frame, f"{speed:.1f} km/h", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(frame, f"{distance:.1f} m", (foot_x, foot_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            output_frames.append(frame)

        return output_frames
