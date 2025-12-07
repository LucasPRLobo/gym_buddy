# training/train_fitness_model.py

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder

from gymbuddy_engine.analysis.execution_sample import ExecutionSample
from gymbuddy_engine.analysis.general_features import extract_generic_features


def build_feature_matrix(samples: list[ExecutionSample]):
    X = []
    y_exercise = []
    y_quality_cls = []

    for sample in samples:
        feats = extract_generic_features(sample.pose_sequence)
        X.append(list(feats.values()))

        lbl = sample.label
        y_exercise.append(lbl.exercise)
        # e.g. 0 = bad, 1 = good (or 0/1/2 for poor/ok/good)
        y_quality_cls.append(lbl.quality_class)

    X = np.array(X)
    le_ex = LabelEncoder()
    y_ex_idx = le_ex.fit_transform(y_exercise)
    y_quality_cls = np.array(y_quality_cls)

    feature_names = list(feats.keys())
    return X, y_ex_idx, y_quality_cls, feature_names, le_ex


def train_fitness_model(samples: list[ExecutionSample], model_path: str):
    X, y_ex, y_qc, feature_names, le_ex = build_feature_matrix(samples)

    # Exercise classifier
    clf_ex = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=300,
        random_state=42,
    )
    clf_ex.fit(X, y_ex)

    # Quality classifier (good/bad or 3 classes)
    clf_q = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=300,
        random_state=42,
    )
    clf_q.fit(X, y_qc)

    joblib.dump(
        {
            "exercise_model": clf_ex,
            "quality_model": clf_q,
            "feature_names": feature_names,
            "exercise_label_encoder": le_ex,
        },
        model_path,
    )
    print(f"Fitness model saved to {model_path}")
