from typing import Dict

from utils import get_center_of_bbox, measure_distance
from config import MAX_PLAYER_BALL_DISTANCE


class PlayerBallAssigner:
    """Assigns the ball to the nearest eligible player each frame."""

    def __init__(self, max_distance: int = MAX_PLAYER_BALL_DISTANCE) -> None:
        self.max_player_ball_distance = max_distance

    def assign_ball_to_player(self, players: Dict, ball_bbox: list) -> int:
        """Return the player_id of the closest player to the ball, or -1 if none."""
        ball_center = get_center_of_bbox(ball_bbox)

        minimum_distance = float("inf")
        assigned_player = -1

        for player_id, player in players.items():
            bbox = player["bbox"]
            # Measure from both bottom corners of the player box to the ball.
            dist_left = measure_distance((bbox[0], bbox[3]), ball_center)
            dist_right = measure_distance((bbox[2], bbox[3]), ball_center)
            distance = min(dist_left, dist_right)

            if distance < self.max_player_ball_distance and distance < minimum_distance:
                minimum_distance = distance
                assigned_player = player_id

        return assigned_player
