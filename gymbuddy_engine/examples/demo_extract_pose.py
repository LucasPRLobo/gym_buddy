# gymbuddy_engine/examples/demo_extract_pose.py

from pathlib import Path

from gymbuddy_engine.ingestion.video_io import load_video_frames
from gymbuddy_engine.pose.pose_estimator import PoseEstimator
from gymbuddy_engine.visualization.overlay import generate_overlay_video


def demo_extract_pose(video_path: str):
    # 1. Load frames from video
    frames, fps = load_video_frames(video_path, target_fps=30)

    print(f"Loaded {len(frames)} frames at ~{fps:.2f} fps")

    # 2. Extract pose sequence
    estimator = PoseEstimator(model_complexity=1)
    pose_seq = estimator.extract_pose_from_frames(
        frames=frames,
        fps=fps,
        exercise="squat",
    )

    print("Exercise:", pose_seq.exercise)
    print("FPS:", pose_seq.fps)
    print("Total frames with pose:", len(pose_seq.frames))

    if pose_seq.frames:
        print("Keypoints in first frame:")
        for name, (x, y) in list(pose_seq.frames[0].keypoints.items())[:10]:
            print(f"  {name}: ({x:.3f}, {y:.3f})")

    # 3. Generate overlay video
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "squat_overlay.mp4"

    print("Generating overlay video at:", output_path)
    generate_overlay_video(frames, pose_seq, str(output_path))

    print("Done.")
    return pose_seq


if __name__ == "__main__":
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[2]
    video_path = project_root / "videos" / "squat_demo.mp4"

    print("Using video:", video_path)
    demo_extract_pose(str(video_path))
