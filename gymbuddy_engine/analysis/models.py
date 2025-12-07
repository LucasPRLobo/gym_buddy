# gymbuddy_engine/analysis/models.py

from dataclasses import dataclass
from typing import Dict, Tuple, List

Point2D = Tuple[float, float]  # (x, y) normalized between 0 and 1


@dataclass
class PoseFrame:
    """
    Represents pose keypoints for a single video frame.
    """
    frame_index: int                # index of the frame in the sequence
    timestamp: float                # time in seconds
    keypoints: Dict[str, Point2D]   # e.g. {"LEFT_KNEE": (0.52, 0.9), ...}


@dataclass
class PoseSequence:
    """
    Represents a full sequence of pose frames for one exercise execution.
    """
    exercise: str                   # e.g. "squat"
    fps: float                      # frames per second
    frames: List[PoseFrame]
