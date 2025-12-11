# training/quick_train_fitness_demo.py

from pathlib import Path

from gymbuddy_engine.ingestion.video_io import load_video_frames
from gymbuddy_engine.pose.pose_estimator import PoseEstimator
from gymbuddy_engine.analysis.execution_sample import ExecutionSample, ExecutionLabel
from training.train_fitness_model import train_fitness_model


def main():
    # Project root = .../gym_buddy
    project_root = Path(__file__).resolve().parents[1]

    # Use the same squat video you used in the demos
    video_path = project_root / "videos" / "squat_demo.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    print(f"Using video for dummy training: {video_path}")

    # 1. Load frames
    frames, fps = load_video_frames(str(video_path), target_fps=30)
    print(f"Loaded {len(frames)} frames at ~{fps:.2f} fps")

    # 2. Extract pose sequence
    estimator = PoseEstimator(model_complexity=1)
    pose_seq = estimator.extract_pose_from_frames(
        frames=frames,
        fps=fps,
        exercise="unknown",  # model will classify later
    )

    # 3. Build some DUMMY samples
    #
    # We need:
    #   - at least 2 exercise classes for the exercise classifier
    #   - at least 2 quality classes for the quality classifier
    #
    # For now we fake this using the SAME pose sequence with different labels.
    # This is ONLY to test the pipeline – not a real model.

    samples: list[ExecutionSample] = []

    # Good squat
    samples.append(
        ExecutionSample(
            pose_sequence=pose_seq,
            label=ExecutionLabel(
                exercise="squat",
                quality_score=85.0,
                is_good_form=True,
                quality_class=2,  # e.g. 0=bad, 1=ok, 2=good
            ),
        )
    )

    # Okay squat
    samples.append(
        ExecutionSample(
            pose_sequence=pose_seq,
            label=ExecutionLabel(
                exercise="squat",
                quality_score=70.0,
                is_good_form=True,
                quality_class=1,
            ),
        )
    )

    # Fake "lunge" using same motion (just for the classifier to have 2 classes)
    samples.append(
        ExecutionSample(
            pose_sequence=pose_seq,
            label=ExecutionLabel(
                exercise="lunge",
                quality_score=60.0,
                is_good_form=False,
                quality_class=1,
            ),
        )
    )

    # Fake "bad lunge"
    samples.append(
        ExecutionSample(
            pose_sequence=pose_seq,
            label=ExecutionLabel(
                exercise="lunge",
                quality_score=30.0,
                is_good_form=False,
                quality_class=0,
            ),
        )
    )

    # 4. Train and save model
    model_path = project_root / "models" / "fitness_model.pkl"
    model_path.parent.mkdir(exist_ok=True)

    print(f"Training dummy fitness model, will save to: {model_path}")
    train_fitness_model(samples, str(model_path))
    print("Done training dummy model.")


if __name__ == "__main__":
    main()
