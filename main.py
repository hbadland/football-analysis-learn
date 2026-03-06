"""
Football analysis pipeline.

Usage:
    python main.py

Adjust INPUT_VIDEO, OUTPUT_VIDEO and MODEL_PATH below (or convert to CLI args).
By default, tracking stubs are read from disk so expensive re-detection is
skipped on repeated runs.  Delete the stubs/ files to force a fresh run.
"""

import logging
import numpy as np

from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — edit these paths as needed
# ---------------------------------------------------------------------------
INPUT_VIDEO = "input_videos/08fd33_4.mp4"
OUTPUT_VIDEO = "output_videos/output.avi"
MODEL_PATH = "models/best.pt"

USE_STUBS = True  # Set to False to re-run detection from scratch.


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load video
    # ------------------------------------------------------------------
    logger.info("Reading video: %s", INPUT_VIDEO)
    frames, fps = read_video(INPUT_VIDEO)

    # ------------------------------------------------------------------
    # 2. Detect & track objects
    # ------------------------------------------------------------------
    tracker = Tracker(MODEL_PATH)
    tracks = tracker.get_object_tracks(
        frames,
        read_from_stub=USE_STUBS,
        stub_path="stubs/track_stubs.pkl",
    )
    tracker.add_position_to_tracks(tracks)

    # ------------------------------------------------------------------
    # 3. Estimate camera movement and correct positions
    # ------------------------------------------------------------------
    camera_estimator = CameraMovementEstimator(frames[0])
    camera_movement = camera_estimator.get_camera_movement(
        frames,
        read_from_stub=USE_STUBS,
        stub_path="stubs/camera_movement_stub.pkl",
    )
    camera_estimator.add_adjusted_positions_to_tracks(tracks, camera_movement)

    # ------------------------------------------------------------------
    # 4. Perspective-transform positions to real-world metres
    # ------------------------------------------------------------------
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # ------------------------------------------------------------------
    # 5. Interpolate missing ball detections
    # ------------------------------------------------------------------
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # ------------------------------------------------------------------
    # 6. Speed and distance
    # ------------------------------------------------------------------
    speed_estimator = SpeedAndDistanceEstimator(frame_rate=fps)
    speed_estimator.add_speed_and_distance_to_tracks(tracks)

    # ------------------------------------------------------------------
    # 7. Assign players to teams
    # ------------------------------------------------------------------
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frames[0], tracks["players"][0])

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track_info in player_track.items():
            team_id = team_assigner.get_player_team(frames[frame_num], track_info["bbox"], player_id)
            tracks["players"][frame_num][player_id]["team"] = team_id
            tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[team_id]

    # ------------------------------------------------------------------
    # 8. Assign ball to nearest player each frame
    # ------------------------------------------------------------------
    ball_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num].get(1, {}).get("bbox")
        if ball_bbox:
            assigned_player = ball_assigner.assign_ball_to_player(player_track, ball_bbox)
            if assigned_player != -1:
                tracks["players"][frame_num][assigned_player]["has_ball"] = True
                team_ball_control.append(tracks["players"][frame_num][assigned_player]["team"])
                continue
        # No assignment — carry forward the last known team in control.
        team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)

    team_ball_control = np.array(team_ball_control)

    # ------------------------------------------------------------------
    # 9. Draw annotations
    # ------------------------------------------------------------------
    output_frames = tracker.draw_annotations(frames, tracks, team_ball_control)
    output_frames = camera_estimator.draw_camera_movement(output_frames, camera_movement)
    output_frames = speed_estimator.draw_speed_and_distance(output_frames, tracks)

    # ------------------------------------------------------------------
    # 10. Save output
    # ------------------------------------------------------------------
    save_video(output_frames, OUTPUT_VIDEO, fps=fps)
    logger.info("Done. Output saved to %s", OUTPUT_VIDEO)


if __name__ == "__main__":
    main()
