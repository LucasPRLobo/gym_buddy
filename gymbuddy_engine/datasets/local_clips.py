import json
from pathlib import Path
from typing import List

from gymbuddy_engine.ingestion.video_io import load_video_frames
from gymbuddy_engine.pose.pose_estimator import PoseEstimator
from gymbuddy_engine.analysis.execution_sample import ExecutionSample, ExecutionLabel


def load_local_clips_dataset(
    root_dir: str,
    target_fps: float = 30.0,
) -> List[ExecutionSample]:
    """
    Loads videos + labels.json from ANY directory.
    Expected directory structure:

      root_dir/
         labels.json
         *.mp4

    labels.json maps:
        "filename.mp4": { "exercise": "squat", "quality_class": 2 }
    """
    root = Path(root_dir)
    labels_path = root / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.json not found in {root_dir}")

    with open(labels_path, "r") as f:
        labels_data = json.load(f)

    estimator = PoseEstimator(model_complexity=1)
    samples: List[ExecutionSample] = []

    for filename, meta in labels_data.items():
        video_path = root / filename
        if not video_path.exists():
            print(f"[WARN] Video not found: {video_path}")
            continue

        print(f"Processing {video_path} → exercise={meta['exercise']} class={meta['quality_class']}")
        
        frames, fps = load_video_frames(str(video_path), target_fps=target_fps)

        pose_seq = estimator.extract_pose_from_frames(
            frames=frames,
            fps=fps,
            exercise=meta["exercise"],
        )

        label = ExecutionLabel(
            exercise=meta["exercise"],
            quality_score=None,
            is_good_form=None,
            quality_class=int(meta["quality_class"]),
        )

        samples.append(ExecutionSample(pose_sequence=pose_seq, label=label))

    return samples
