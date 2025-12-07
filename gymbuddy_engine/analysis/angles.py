# gymbuddy_engine/analysis/angles.py

from typing import Tuple
import math
from gymbuddy_engine.analysis.models import PoseFrame, Point2D


def _angle_between(p1: Point2D, p2: Point2D, p3: Point2D) -> float:
    """
    Compute the angle at point p2 formed by (p1 -> p2 -> p3), in degrees.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    v1 = (x1 - x2, y1 - y2)
    v2 = (x3 - x2, y3 - y2)

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    norm1 = math.hypot(*v1)
    norm2 = math.hypot(*v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0

    cos_theta = max(-1.0, min(1.0, dot / (norm1 * norm2)))
    angle = math.degrees(math.acos(cos_theta))
    return angle


def knee_angle(frame: PoseFrame, side: str = "LEFT") -> float:
    """
    Approximate knee angle (hip-knee-ankle) in degrees.
    side: "LEFT" or "RIGHT"
    """
    hip = frame.keypoints.get(f"{side}_HIP")
    knee = frame.keypoints.get(f"{side}_KNEE")
    ankle = frame.keypoints.get(f"{side}_ANKLE")
    if hip is None or knee is None or ankle is None:
        return 0.0
    return _angle_between(hip, knee, ankle)


def hip_angle(frame: PoseFrame, side: str = "LEFT") -> float:
    """
    Approximate hip flexion angle (shoulder-hip-knee) in degrees.
    side: "LEFT" or "RIGHT"
    """
    shoulder = frame.keypoints.get(f"{side}_SHOULDER")
    hip = frame.keypoints.get(f"{side}_HIP")
    knee = frame.keypoints.get(f"{side}_KNEE")
    if shoulder is None or hip is None or knee is None:
        return 0.0
    return _angle_between(shoulder, hip, knee)


def torso_lean_angle(frame: PoseFrame) -> float:
    """
    Angle of the line between hips and shoulders relative to vertical.
    Rough proxy for torso lean.
    """
    ls = frame.keypoints.get("LEFT_SHOULDER")
    rs = frame.keypoints.get("RIGHT_SHOULDER")
    lh = frame.keypoints.get("LEFT_HIP")
    rh = frame.keypoints.get("RIGHT_HIP")

    if ls is None or rs is None or lh is None or rh is None:
        return 0.0

    shoulder_center = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    hip_center = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)

    dx = shoulder_center[0] - hip_center[0]
    dy = shoulder_center[1] - hip_center[1]

    # Vertical vector is (0, -1); angle between (dx, dy) and vertical
    dot = dx * 0 + dy * -1
    norm = math.hypot(dx, dy)
    if norm == 0:
        return 0.0

    cos_theta = max(-1.0, min(1.0, dot / norm))
    angle = math.degrees(math.acos(cos_theta))
    return angle
