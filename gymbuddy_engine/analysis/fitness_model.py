# gymbuddy_engine/analysis/fitness_model.py

from dataclasses import dataclass
from typing import List

import joblib
import numpy as np

from gymbuddy_engine.analysis.models import PoseSequence
from gymbuddy_engine.analysis.general_features import extract_generic_features


@dataclass
class FitnessPrediction:
    """
    Output of the multi-exercise fitness model.
    """
    exercise: str
    exercise_confidence: float
    quality_class: int         # e.g. 0 = bad, 1 = ok, 2 = good
    quality_confidence: float


class FitnessModel:
    """
    Multi-exercise model wrapper.

    Expects a joblib file with:
        {
            "exercise_model": clf_ex,
            "quality_model": clf_q,
            "feature_names": [...],
            "exercise_label_encoder": LabelEncoder(...)
        }
    saved by training/train_fitness_model.py
    """

    def __init__(self, model_path: str):
        data = joblib.load(model_path)
        self.exercise_model = data["exercise_model"]
        self.quality_model = data["quality_model"]
        self.feature_names = data["feature_names"]
        self.exercise_label_encoder = data["exercise_label_encoder"]

    def analyze(self, pose_seq: PoseSequence) -> FitnessPrediction:
        """
        Run exercise classification + quality classification on a PoseSequence.
        """
        feats = extract_generic_features(pose_seq)

        # Turn feature dict into ordered vector
        X = np.array([[feats[name] for name in self.feature_names]])

        # ---- Exercise prediction ----
        probs_ex = self.exercise_model.predict_proba(X)[0]
        idx_ex = int(probs_ex.argmax())
        exercise = self.exercise_label_encoder.inverse_transform([idx_ex])[0]
        ex_conf = float(probs_ex[idx_ex])

        # ---- Quality prediction ----
        probs_q = self.quality_model.predict_proba(X)[0]
        idx_q = int(probs_q.argmax())
        q_conf = float(probs_q[idx_q])

        return FitnessPrediction(
            exercise=exercise,
            exercise_confidence=ex_conf,
            quality_class=idx_q,
            quality_confidence=q_conf,
        )
