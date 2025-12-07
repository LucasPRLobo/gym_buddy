# gymbuddy_engine/analysis/execution_sample.py

from dataclasses import dataclass
from typing import List, Optional
from gymbuddy_engine.analysis.models import PoseSequence


@dataclass
class ExecutionLabel:
    exercise: str                   # e.g. "squat", "pushup", "row"
    quality_score: Optional[float]  # e.g. 0-100 (can be None if unknown)
    is_good_form: Optional[bool]    # True/False or None if not annotated
    # optional: coarse quality classes (e.g. 0,1,2)
    quality_class: Optional[int] = None


@dataclass
class ExecutionSample:
    pose_sequence: PoseSequence
    label: Optional[ExecutionLabel] = None
    metadata: Optional[dict] = None
