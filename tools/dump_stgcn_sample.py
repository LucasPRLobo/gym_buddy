import cv2
import numpy as np

def extract_stgcn_sample(video_path: str, T: int = 60, V: int = 33):
    """
    Returns:
      X: np.ndarray shape (C=4, T, V, M=1) with channels (x,y,z,vis)
    """
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise RuntimeError("No frames read from video.")

    # Uniformly sample/resize to exactly T frames
    idx = np.linspace(0, len(frames) - 1, T).astype(int)

    # X will be (C, T, V, M=1)
    X = np.zeros((4, T, V, 1), dtype=np.float32)

    for t, i in enumerate(idx):
        frame = frames[i]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks is None:
            # leave zeros (or you could forward-fill last pose)
            continue

        for v, lm in enumerate(res.pose_landmarks.landmark):
            X[0, t, v, 0] = lm.x           # normalized [0..1] (image coords)
            X[1, t, v, 0] = lm.y           # normalized [0..1]
            X[2, t, v, 0] = lm.z           # roughly normalized depth
            X[3, t, v, 0] = lm.visibility  # [0..1]

    pose.close()
    return X

if __name__ == "__main__":
    video = "/home/lucas-lobo/Programing/gym_buddy/videos/squat_demo.mp4"
    X = extract_stgcn_sample(video, T=60)

    print("ST-GCN sample tensor shape:", X.shape)  # (4, 60, 33, 1)
    # Show a tiny slice: first 3 frames, first 5 joints, all 4 channels
    # output shape will be (C, 3, 5, 1)
    print("Slice (C, t=0..2, v=0..4, M=0):\n", X[:, :3, :5, 0])
