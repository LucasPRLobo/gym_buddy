# gymbuddy_engine/ingestion/video_io.py

import cv2
from typing import List, Tuple


def load_video_frames(
    video_path: str,
    max_frames: int | None = None,
    target_fps: float | None = None
) -> Tuple[List, float]:
    """
    Load frames from a video file.

    Args:
        video_path: Path to the video file.
        max_frames: Optional maximum number of frames to load.
        target_fps: Optional target FPS. If set, frames will be subsampled
                    from the original FPS to approximate this value.

    Returns:
        frames: List of frames as numpy arrays (H, W, 3) in BGR (OpenCV format).
        effective_fps: The effective FPS after subsampling.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_index = 0

    # Compute step for subsampling to target_fps (if requested).
    if target_fps is not None and target_fps > 0:
        step = int(round(orig_fps / target_fps))
        step = max(step, 1)
        effective_fps = orig_fps / step
    else:
        step = 1
        effective_fps = orig_fps

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Keep only 1 out of every "step" frames.
        if frame_index % step == 0:
            frames.append(frame)

            if max_frames is not None and len(frames) >= max_frames:
                break

        frame_index += 1

    cap.release()
    return frames, effective_fps
