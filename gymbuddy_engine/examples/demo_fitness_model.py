# gymbuddy_engine/examples/demo_fitness_model.py

from pathlib import Path

from gymbuddy_engine.ingestion.video_io import load_video_frames
from gymbuddy_engine.pose.pose_estimator import PoseEstimator
from gymbuddy_engine.visualization.overlay import generate_overlay_video
from gymbuddy_engine.analysis.fitness_model import FitnessModel


def demo_fitness(video_path: str, model_path: str):
    # 1. Load frames
    frames, fps = load_video_frames(video_path, target_fps=30)
    print(f"Loaded {len(frames)} frames at ~{fps:.2f} fps")

    # 2. Extract pose sequence
    estimator = PoseEstimator(model_complexity=1)
    pose_seq = estimator.extract_pose_from_frames(
        frames=frames,
        fps=fps,
        exercise="unknown",  # model will classify exercise
    )

    # 3. Load fitness model
    fitness_model = FitnessModel(model_path=model_path)

    # 4. Analyze
    pred = fitness_model.analyze(pose_seq)

    print("\n=== Fitness model prediction ===")
    print(f"Detected exercise: {pred.exercise} (conf: {pred.exercise_confidence:.2f})")
    print(f"Quality class: {pred.quality_class} (conf: {pred.quality_confidence:.2f})")

    # 5. Generate overlay video
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "fitness_overlay.mp4"

    print("\nGenerating overlay video at:", output_path)
    generate_overlay_video(frames, pose_seq, str(output_path), fps=fps)
    print("Done.")

    return pred, str(output_path)


if __name__ == "__main__":
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[2]

    video_path = project_root / "videos" / "squat_demo.mp4"   # or any exercise video
    model_path = project_root / "models" / "fitness_model.pkl"

    print("Using video:", video_path)
    print("Using model:", model_path)

    demo_fitness(str(video_path), str(model_path))
