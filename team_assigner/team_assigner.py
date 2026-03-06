import logging
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class TeamAssigner:
    """Assigns players to teams by clustering jersey colours with K-means."""

    def __init__(self) -> None:
        self.team_colors: Dict[int, np.ndarray] = {}
        self.player_team_dict: Dict[int, int] = {}
        self.kmeans: Optional[KMeans] = None

    # ------------------------------------------------------------------
    # Colour extraction
    # ------------------------------------------------------------------

    def _get_clustering_model(self, image: np.ndarray) -> KMeans:
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=0)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        """Return the dominant jersey colour for a player bounding box."""
        x1, y1, x2, y2 = (int(v) for v in bbox)
        crop = frame[y1:y2, x1:x2]
        top_half = crop[: crop.shape[0] // 2, :]

        kmeans = self._get_clustering_model(top_half)
        labels = kmeans.labels_.reshape(top_half.shape[0], top_half.shape[1])

        # The background cluster is whichever label appears most in the corners.
        corner_labels = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
        background_cluster = max(set(corner_labels), key=corner_labels.count)
        player_cluster = 1 - background_cluster

        return kmeans.cluster_centers_[player_cluster]

    # ------------------------------------------------------------------
    # Team colour calibration
    # ------------------------------------------------------------------

    def assign_team_color(self, frame: np.ndarray, player_detections: Dict) -> None:
        """Fit the two-team K-means model from the first frame's player crops."""
        player_colors = [
            self.get_player_color(frame, det["bbox"])
            for det in player_detections.values()
        ]

        if len(player_colors) < 2:
            logger.warning("Fewer than 2 players detected — team colour assignment may be unreliable.")

        self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=0)
        self.kmeans.fit(player_colors)

        self.team_colors[1] = self.kmeans.cluster_centers_[0]
        self.team_colors[2] = self.kmeans.cluster_centers_[1]
        logger.info("Team colours: %s, %s", self.team_colors[1], self.team_colors[2])

    # ------------------------------------------------------------------
    # Per-player assignment
    # ------------------------------------------------------------------

    def get_player_team(self, frame: np.ndarray, player_bbox: list, player_id: int) -> int:
        """Return 1 or 2 for the team this player belongs to."""
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        if self.kmeans is None:
            raise RuntimeError("Call assign_team_color() before get_player_team().")

        player_color = self.get_player_color(frame, player_bbox)
        team_id = int(self.kmeans.predict(player_color.reshape(1, -1))[0]) + 1

        self.player_team_dict[player_id] = team_id
        return team_id
