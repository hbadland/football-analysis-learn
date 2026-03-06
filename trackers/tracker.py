import logging
import os
import pickle
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO

from utils import get_center_of_bbox, get_bbox_width, get_foot_position
from config import DETECTION_CONF_THRESHOLD, DETECTION_BATCH_SIZE

logger = logging.getLogger(__name__)

Tracks = Dict[str, List[Dict]]


class Tracker:
    def __init__(self, model_path: str) -> None:
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    # ------------------------------------------------------------------
    # Detection & tracking
    # ------------------------------------------------------------------

    def detect_frames(self, frames: List[np.ndarray]) -> list:
        detections = []
        for i in range(0, len(frames), DETECTION_BATCH_SIZE):
            batch = self.model.predict(frames[i : i + DETECTION_BATCH_SIZE], conf=DETECTION_CONF_THRESHOLD)
            detections += batch
        return detections

    def get_object_tracks(
        self,
        frames: List[np.ndarray],
        read_from_stub: bool = False,
        stub_path: Optional[str] = None,
    ) -> Tracks:
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)

        tracks: Tracks = {"players": [], "referees": [], "ball": []}

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            det_sv = sv.Detections.from_ultralytics(detection)

            # Reclassify goalkeepers as players so they are tracked together.
            if "goalkeeper" in cls_names_inv:
                for idx, class_id in enumerate(det_sv.class_id):
                    if cls_names[class_id] == "goalkeeper":
                        det_sv.class_id[idx] = cls_names_inv["player"]

            det_with_tracks = self.tracker.update_with_detections(det_sv)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_det in det_with_tracks:
                bbox = frame_det[0].tolist()
                cls_id = frame_det[3]
                track_id = frame_det[4]

                if cls_id == cls_names_inv.get("player"):
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                elif cls_id == cls_names_inv.get("referee"):
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_det in det_sv:
                bbox = frame_det[0].tolist()
                cls_id = frame_det[3]
                if cls_id == cls_names_inv.get("ball"):
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)
            logger.info("Saved tracking stubs to %s", stub_path)

        return tracks

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------

    def add_position_to_tracks(self, tracks: Tracks) -> None:
        for object_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info["bbox"]
                    position = (
                        get_center_of_bbox(bbox)
                        if object_name == "ball"
                        else get_foot_position(bbox)
                    )
                    tracks[object_name][frame_num][track_id]["position"] = position

    def interpolate_ball_positions(self, ball_positions: List[Dict]) -> List[Dict]:
        raw = [x.get(1, {}).get("bbox", []) for x in ball_positions]
        df = pd.DataFrame(raw, columns=["x1", "y1", "x2", "y2"])
        df = df.interpolate().bfill()
        return [{1: {"bbox": row}} for row in df.to_numpy().tolist()]

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_ellipse(
        self,
        frame: np.ndarray,
        bbox: list,
        color: tuple,
        track_id: Optional[int] = None,
    ) -> np.ndarray:
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        if track_id is not None:
            rect_w, rect_h = 40, 20
            x1_rect = x_center - rect_w // 2
            x2_rect = x_center + rect_w // 2
            y1_rect = y2 - rect_h // 2 + 15
            y2_rect = y2 + rect_h // 2 + 15

            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED)

            text_x = int(x1_rect) + (2 if track_id > 99 else 12)
            cv2.putText(
                frame,
                str(track_id),
                (text_x, int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        return frame

    def _draw_triangle(self, frame: np.ndarray, bbox: list, color: tuple) -> np.ndarray:
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        triangle_points = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
        return frame

    def _draw_team_ball_control(
        self,
        frame: np.ndarray,
        frame_num: int,
        team_ball_control: np.ndarray,
    ) -> np.ndarray:
        h, w = frame.shape[:2]

        # Semi-transparent overlay box — positioned relative to frame size.
        box_x1 = int(w * 0.70)
        box_y1 = int(h * 0.88)
        box_x2 = w - 10
        box_y2 = h - 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        control_so_far = team_ball_control[: frame_num + 1]
        team1_frames = int((control_so_far == 1).sum())
        team2_frames = int((control_so_far == 2).sum())
        total = team1_frames + team2_frames or 1

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Team 1 Ball Control: {team1_frames / total * 100:.1f}%", (box_x1 + 10, box_y1 + 30), font, 0.8, (0, 0, 0), 2)
        cv2.putText(frame, f"Team 2 Ball Control: {team2_frames / total * 100:.1f}%", (box_x1 + 10, box_y1 + 60), font, 0.8, (0, 0, 0), 2)

        return frame

    def draw_annotations(
        self,
        video_frames: List[np.ndarray],
        tracks: Tracks,
        team_ball_control: np.ndarray,
    ) -> List[np.ndarray]:
        from config import COLOUR_REFEREE, COLOUR_BALL, COLOUR_BALL_POSSESSION

        output_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            for track_id, player in tracks["players"][frame_num].items():
                color = player.get("team_color", (0, 0, 255))
                frame = self._draw_ellipse(frame, player["bbox"], color, track_id)
                if player.get("has_ball"):
                    frame = self._draw_triangle(frame, player["bbox"], COLOUR_BALL_POSSESSION)

            for _, referee in tracks["referees"][frame_num].items():
                frame = self._draw_ellipse(frame, referee["bbox"], COLOUR_REFEREE)

            for _, ball in tracks["ball"][frame_num].items():
                frame = self._draw_triangle(frame, ball["bbox"], COLOUR_BALL)

            frame = self._draw_team_ball_control(frame, frame_num, team_ball_control)
            output_frames.append(frame)

        return output_frames
