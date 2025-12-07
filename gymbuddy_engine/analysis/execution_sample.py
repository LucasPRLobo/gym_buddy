# gymbuddy_engine/analysis/execution_sample.py

from dataclasses import dataclass
from typing import List, Dict, Optional
from gymbuddy_engine.analysis.models import PoseSequence


@dataclass
class ExecutionLabel:
    """
    Labels describing the quality of a single exercise execution.
    """
    exercise: str                 # e.g. "squat"
    is_good_form: bool            # True if overall good
    score: Optional[float] = None # 0-100, if available
    error_tags: Optional[List[str]] = None  # e.g. ["depth_insufficient", "knee_valgus"]


@dataclass
class ExecutionSample:
    """
    A single example used for training or inference.
    """
    pose_sequence: PoseSequence
    label: Optional[ExecutionLabel] = None  # None during inference
    metadata: Optional[Dict] = None         # dataset_name, subject_id, etc.
