from typing import Optional, Tuple

import cv2
import numpy as np

from config import PITCH_WIDTH_M, PITCH_LENGTH_M, PITCH_PIXEL_VERTICES


class ViewTransformer:
    """Maps pixel coordinates to real-world pitch coordinates (metres).

    Uses a perspective (homography) transform derived from four known
    pitch corner points visible in the camera frame.
    """

    def __init__(
        self,
        pixel_vertices: Optional[list] = None,
        pitch_width_m: float = PITCH_WIDTH_M,
        pitch_length_m: float = PITCH_LENGTH_M,
    ) -> None:
        if pixel_vertices is None:
            pixel_vertices = PITCH_PIXEL_VERTICES

        self.pitch_width = pitch_width_m
        self.pitch_length = pitch_length_m

        # Real-world target rectangle (top-left, top-right, bottom-right, bottom-left).
        target_vertices = np.array(
            [
                [0, pitch_width_m],
                [0, 0],
                [pitch_length_m, 0],
                [pitch_length_m, pitch_width_m],
            ],
            dtype=np.float32,
        )

        src = np.array(pixel_vertices, dtype=np.float32)
        self._transform_matrix = cv2.getPerspectiveTransform(src, target_vertices)

        # Polygon for point-in-bounds test.
        self._pixel_polygon = src.reshape((-1, 1, 2)).astype(np.float32)

    # ------------------------------------------------------------------

    def transform_point(self, point: Tuple[float, float]) -> Optional[np.ndarray]:
        """Convert a single pixel (x, y) to pitch coordinates (m).

        Returns None if the point is outside the defined pitch region.
        """
        px = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
        if cv2.pointPolygonTest(self._pixel_polygon, (float(point[0]), float(point[1])), False) < 0:
            return None

        transformed = cv2.perspectiveTransform(px, self._transform_matrix)
        return transformed[0][0]

    def add_transformed_position_to_tracks(self, tracks: dict) -> None:
        """Add 'position_transformed' (metres) to every track entry."""
        for object_tracks in tracks.values():
            for frame_track in object_tracks:
                for track_id, track_info in frame_track.items():
                    position = track_info.get("position_adjusted") or track_info.get("position")
                    if position is None:
                        continue
                    transformed = self.transform_point(position)
                    if transformed is not None:
                        track_info["position_transformed"] = transformed.tolist()
