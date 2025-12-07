# gymbuddy_engine/visualization/overlay.py

from typing import List, Tuple, Dict
import cv2
import numpy as np

from gymbuddy_engine.analysis.models import PoseSequence, PoseFrame, Point2D


# Define which keypoints to connect to form the skeleton.
# These names must match MediaPipe's PoseLandmark names.
SKELETON_CONNECTIONS: List[Tuple[str, str]] = [
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_HIP", "LEFT_KNEE"),
    ("LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_HIP", "RIGHT_KNEE"),
    ("RIGHT_KNEE", "RIGHT_ANKLE"),
]


def _denormalize_point(
    point: Point2D,
    width: int,
    height: int,
) -> Tuple[int, int]:
    """
    Convert normalized (x, y) in [0,1] range to pixel coordinates.
    """
    x_norm, y_norm = point
    x = int(x_norm * width)
    y = int(y_norm * height)
    return x, y


def draw_pose_on_frame(
    frame_bgr: np.ndarray,
    pose_frame: PoseFrame,
    point_radius: int = 4,
    point_thickness: int = -1,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Draw skeleton keypoints and connections on a single frame.

    Args:
        frame_bgr: Original frame (BGR, OpenCV format).
        pose_frame: PoseFrame with keypoints for this frame.
        point_radius: Radius of keypoint circles.
        point_thickness: Thickness of keypoint circles (-1 = filled).
        line_thickness: Thickness of skeleton lines.

    Returns:
        A copy of the frame with the skeleton overlay.
    """
    output = frame_bgr.copy()
    h, w, _ = output.shape

    # Colors in BGR
    point_color = (0, 255, 0)   # green for joints
    line_color = (255, 255, 255)  # white for bones

    # Draw joints
    for name, pt in pose_frame.keypoints.items():
        x, y = _denormalize_point(pt, w, h)
        cv2.circle(output, (x, y), point_radius, point_color, point_thickness)

    # Draw skeleton connections
    kps: Dict[str, Point2D] = pose_frame.keypoints
    for start_name, end_name in SKELETON_CONNECTIONS:
        if start_name in kps and end_name in kps:
            x1, y1 = _denormalize_point(kps[start_name], w, h)
            x2, y2 = _denormalize_point(kps[end_name], w, h)
            cv2.line(output, (x1, y1), (x2, y2), line_color, line_thickness)

    return output


def generate_overlay_video(
    frames: List[np.ndarray],
    pose_seq: PoseSequence,
    output_path: str,
    fps: float | None = None,
) -> str:
    """
    Generate a video with skeleton overlay drawn on each frame.

    Args:
        frames: List of original frames (BGR).
        pose_seq: PoseSequence with one PoseFrame per original frame index.
        output_path: Path where the output video will be saved.
        fps: Optional FPS. If None, uses pose_seq.fps.

    Returns:
        output_path: The path to the generated video file.
    """
    if fps is None:
        fps = pose_seq.fps

    if not frames:
        raise ValueError("No frames provided to generate_overlay_video.")

    height, width, _ = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Build a quick lookup from frame_index -> PoseFrame
    pose_by_index: Dict[int, PoseFrame] = {
        pf.frame_index: pf for pf in pose_seq.frames
    }

    for idx, frame in enumerate(frames):
        pose_frame = pose_by_index.get(idx, None)
        if pose_frame is not None:
            overlay = draw_pose_on_frame(frame, pose_frame)
        else:
            # If we don't have pose data for this frame, just use the original
            overlay = frame

        writer.write(overlay)

    writer.release()
    return output_path
