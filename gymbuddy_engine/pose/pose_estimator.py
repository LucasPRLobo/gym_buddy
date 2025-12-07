# gymbuddy_engine/pose/pose_estimator.py

from typing import List, Dict, Tuple
import cv2
import mediapipe as mp

from gymbuddy_engine.analysis.models import PoseFrame, PoseSequence

mp_pose = mp.solutions.pose


class PoseEstimator:
    """
    Wrapper around MediaPipe Pose to extract pose keypoints from video frames.
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

    def extract_pose_from_frames(
        self,
        frames: List,
        fps: float,
        exercise: str,
    ) -> PoseSequence:
        """
        Extract a PoseSequence from a list of video frames.

        Args:
            frames: List of BGR images (OpenCV format).
            fps: Frames per second of the input sequence.
            exercise: Name of the exercise (e.g. "squat").

        Returns:
            PoseSequence with pose keypoints for each frame.
        """
        pose_frames: List[PoseFrame] = []

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            enable_segmentation=False,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        ) as pose:

            for idx, frame in enumerate(frames):
                # Convert BGR (OpenCV) to RGB for MediaPipe.
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = pose.process(image_rgb)
                keypoints: Dict[str, Tuple[float, float]] = {}

                if results.pose_landmarks:
                    # Landmarks are already in normalized coordinates [0, 1].
                    for lm_id, landmark in enumerate(results.pose_landmarks.landmark):
                        lm_name = mp_pose.PoseLandmark(lm_id).name  # e.g. "LEFT_KNEE"
                        x = landmark.x
                        y = landmark.y
                        keypoints[lm_name] = (x, y)

                timestamp = idx / fps

                pose_frames.append(
                    PoseFrame(
                        frame_index=idx,
                        timestamp=timestamp,
                        keypoints=keypoints,
                    )
                )

        return PoseSequence(
            exercise=exercise,
            fps=fps,
            frames=pose_frames,
        )
