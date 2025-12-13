# gymbuddy_engine/datasets/workout_fitness_hasyim.py

from pathlib import Path
from typing import List

from gymbuddy_engine.ingestion.video_io import load_video_frames
from gymbuddy_engine.pose.pose_estimator import PoseEstimator
from gymbuddy_engine.analysis.execution_sample import ExecutionSample, ExecutionLabel


def load_workout_fitness_hasyim_dataset(
    project_root: str,
    target_fps: float = 30.0,
    max_videos_per_class: int | None = None,
) -> List[ExecutionSample]:
    """
    Load the Kaggle 'Workout/Exercises Video' dataset by hasyimabdillah.

    Expected structure:
        <project_root>/datasets/workout_fitness_hasyim/
            squat/
              *.mp4
            pushup/
              *.mp4
            ...

    Each subfolder name is treated as the exercise label.
    """
    root = Path(project_root) / "datasets" / "workout_fitness_hasyim"
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    estimator = PoseEstimator(model_complexity=1)
    samples: List[ExecutionSample] = []

    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue

        # Folder name is the exercise label (e.g. "squat", "pushup", "pull_up")
        exercise_name = class_dir.name.lower()
        video_files = sorted(class_dir.glob("*.mp4"))

        if max_videos_per_class is not None:
            video_files = video_files[:max_videos_per_class]

        if not video_files:
            continue

        print(f"[Dataset] Exercise '{exercise_name}' in {class_dir}, {len(video_files)} videos")

        for video_path in video_files:
            print(f"  -> Processing {video_path}")
            frames, fps = load_video_frames(str(video_path), target_fps=target_fps)
            if not frames:
                print(f"  [WARN] No frames for {video_path}, skipping.")
                continue

            pose_seq = estimator.extract_pose_from_frames(
                frames=frames,
                fps=fps,
                exercise=exercise_name,
            )

            # No quality labels here – we just set neutral placeholders.
            label = ExecutionLabel(
                exercise=exercise_name,
                quality_score=None,
                is_good_form=None,
                quality_class=1,
            )

            samples.append(ExecutionSample(pose_sequence=pose_seq, label=label))

    return samples
