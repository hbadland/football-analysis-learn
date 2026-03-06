from typing import Tuple


BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2
Point = Tuple[int, int]


def get_center_of_bbox(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox: BBox) -> float:
    return bbox[2] - bbox[0]


def measure_distance(p1: Tuple, p2: Tuple) -> float:
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def measure_xy_distance(p1: Tuple, p2: Tuple) -> Tuple[float, float]:
    return p1[0] - p2[0], p1[1] - p2[1]


def get_foot_position(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)
