# training/train_exercise_classifier_hasyim.py

from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from gymbuddy_engine.datasets.workout_fitness_hasyim import (
    load_workout_fitness_hasyim_dataset,
)
from gymbuddy_engine.analysis.execution_sample import ExecutionSample
from gymbuddy_engine.analysis.general_features import extract_generic_features


def build_feature_matrix(samples: list[ExecutionSample]):
    """
    Convert ExecutionSample list into:
        X: feature matrix (N_samples x N_features)
        y: encoded exercise labels
        feature_names: list of feature keys
        label_encoder: LabelEncoder for exercise names
    """
    X = []
    y_labels = []
    feature_names = None

    for sample in samples:
        feats = extract_generic_features(sample.pose_sequence)

        # Freeze feature ordering on first sample so it's consistent.
        if feature_names is None:
            feature_names = list(feats.keys())

        X.append([feats[name] for name in feature_names])
        y_labels.append(sample.label.exercise)

    X = np.array(X, dtype=np.float32)
    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    return X, y, feature_names, le


def train_exercise_classifier(
    project_root: str,
    target_fps: float = 30.0,
    max_videos_per_class: int | None = None,
):
    project = Path(project_root)

    print("Loading Workout/Fitness dataset (hasyimabdillah)...")
    samples = load_workout_fitness_hasyim_dataset(
        project_root=project_root,
        target_fps=target_fps,
        max_videos_per_class=max_videos_per_class,
    )

    if not samples:
        raise RuntimeError("No samples loaded from dataset. Check dataset path or contents.")

    print(f"Total samples: {len(samples)}")

    X, y, feature_names, label_encoder = build_feature_matrix(samples)

    print("Feature matrix shape:", X.shape)
    print("Number of exercise classes:", len(label_encoder.classes_))
    print("Exercise classes:", list(label_encoder.classes_))

    # Simple train/val split for sanity check
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        max_iter=200,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Quick evaluation
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"\nValidation accuracy: {acc:.3f}\n")
    print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))

    # Save model
    model_path = project / "models" / "exercise_classifier_hasyim.pkl"
    model_path.parent.mkdir(exist_ok=True)

    joblib.dump(
        {
            "exercise_model": clf,
            "feature_names": feature_names,
            "exercise_label_encoder": label_encoder,
        },
        model_path,
    )

    print(f"\n[OK] Exercise classifier saved to {model_path}")


if __name__ == "__main__":
    train_exercise_classifier(
        "/home/lucas-lobo/Programing/gym_buddy",
        target_fps=30.0,
        max_videos_per_class=10,  # reduce for faster first run; increase later
    )
