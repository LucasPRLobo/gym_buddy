# gymbuddy_engine/analysis/general_features.py

from typing import Dict, List
import numpy as np

from gymbuddy_engine.analysis.models import PoseSequence, PoseFrame
from gymbuddy_engine.analysis.angles import _angle_between  # reuse helper


JOINT_TRIPLETS = [
    ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
    ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"),
    ("RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"),
    # add more as needed
]


def _angle_for_triplet(frame: PoseFrame, a: str, b: str, c: str) -> float:
    p1 = frame.keypoints.get(a)
    p2 = frame.keypoints.get(b)
    p3 = frame.keypoints.get(c)
    if p1 is None or p2 is None or p3 is None:
        return 0.0
    return _angle_between(p1, p2, p3)


def extract_generic_features(pose_seq: PoseSequence) -> Dict[str, float]:
    """
    Extract generic joint-angle and symmetry features from any exercise.
    """
    per_triplet_angles: Dict[str, List[float]] = {
        f"{a}_{b}_{c}": [] for (a, b, c) in JOINT_TRIPLETS
    }

    # Collect angle time series
    for frame in pose_seq.frames:
        for (a, b, c) in JOINT_TRIPLETS:
            key = f"{a}_{b}_{c}"
            theta = _angle_for_triplet(frame, a, b, c)
            per_triplet_angles[key].append(theta)

    feats: Dict[str, float] = {}

    # Aggregate statistics: min, max, mean, std, ROM
    for key, series in per_triplet_angles.items():
        if not series:
            feats[f"{key}_min"] = 0.0
            feats[f"{key}_max"] = 0.0
            feats[f"{key}_mean"] = 0.0
            feats[f"{key}_std"] = 0.0
            feats[f"{key}_rom"] = 0.0
            continue

        x = np.array(series)
        feats[f"{key}_min"] = float(x.min())
        feats[f"{key}_max"] = float(x.max())
        feats[f"{key}_mean"] = float(x.mean())
        feats[f"{key}_std"] = float(x.std())
        feats[f"{key}_rom"] = float(x.max() - x.min())

    # Simple temporal feature: number of frames / duration
    feats["num_frames"] = float(len(pose_seq.frames))
    feats["duration_sec"] = float(len(pose_seq.frames) / max(pose_seq.fps, 1e-6))

    return feats
