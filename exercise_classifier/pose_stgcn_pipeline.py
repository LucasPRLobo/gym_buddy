#!/usr/bin/env python3
"""
MediaPipe Pose -> ST-GCN-ready pipeline.

Outputs:
- Annotated video (pose overlay)
- Debug frames (pose overlay)
- Pose tensors: X (33 joints) and X_body (22 joints)
- Diagnostic plots
- Partitioned adjacency A_body (K=3) for body-only graph
- Torch tensors x_in (N,C,T,V,M) and A_body (K,V,V)

Install:
  pip install opencv-python mediapipe numpy matplotlib torch

Run example:
  python pose_stgcn_pipeline.py \
    --video "/path/to/video.mp4" \
    --out_dir "./debug_pose" \
    --t_target 60 \
    --debug_ts 0 10 20 30 40 50 \
    --joint 24
"""

import os
import argparse
from dataclasses import dataclass
from collections import deque

import cv2
import numpy as np
import matplotlib.pyplot as plt

import mediapipe as mp

import torch


# --------------------------
# Config / helpers
# --------------------------

C = 4          # x, y, z, visibility
V_FULL = 33    # MediaPipe Pose landmarks
M = 1          # single person


def read_video_info(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, w, h, n


def uniform_sample_indices(n_frames: int, t_target: int) -> np.ndarray:
    """Uniformly sample indices from [0, n_frames-1] to length t_target."""
    if n_frames <= 0:
        raise ValueError("n_frames must be > 0")
    if t_target <= 0:
        raise ValueError("t_target must be > 0")
    if n_frames == 1:
        return np.zeros((t_target,), dtype=int)
    return np.linspace(0, n_frames - 1, t_target).round().astype(int)


def landmarks_to_tensor(landmarks) -> np.ndarray:
    """
    landmarks: mp.framework.formats.landmark_pb2.NormalizedLandmarkList
    returns: (C, V) float32 -> [x, y, z, visibility] per joint
    """
    arr = np.zeros((C, V_FULL), dtype=np.float32)
    for j, lm in enumerate(landmarks.landmark):
        arr[0, j] = lm.x
        arr[1, j] = lm.y
        arr[2, j] = lm.z
        arr[3, j] = lm.visibility
    return arr


def empty_tensor() -> np.ndarray:
    return np.zeros((C, V_FULL), dtype=np.float32)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# --------------------------
# Graph utilities (ST-GCN)
# --------------------------

def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    """Symmetric normalization: D^{-1/2} A D^{-1/2}."""
    Dl = np.sum(A, axis=1)
    Dn = np.diag(1.0 / np.sqrt(np.maximum(Dl, 1e-6)))
    return (Dn @ A @ Dn).astype(np.float32)


def shortest_hop_dist(V: int, edges: list[tuple[int, int]], center: int) -> np.ndarray:
    """BFS hop distance from center. Returns dist with -1 for unreachable."""
    adj = [[] for _ in range(V)]
    for i, j in edges:
        if not (0 <= i < V and 0 <= j < V):
            raise ValueError(f"Edge ({i},{j}) out of range for V={V}")
        adj[i].append(j)
        adj[j].append(i)

    dist = np.full(V, -1, dtype=np.int32)
    dist[center] = 0
    q = deque([center])

    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)

    return dist


def build_partitioned_adjacency(V: int, neighbor_links: list[tuple[int, int]], center: int) -> torch.Tensor:
    """
    Build the classic ST-GCN spatial configuration partition (K=3):
      - root (same hop distance or self links)
      - centripetal (toward center)
      - centrifugal (away from center)

    Returns:
      A: torch.FloatTensor of shape (K=3, V, V)
    """
    hop = shortest_hop_dist(V, neighbor_links, center=center)
    unreachable = np.where(hop == -1)[0]
    if len(unreachable) > 0:
        raise ValueError(
            f"Graph not connected from center={center}. "
            f"Unreachable nodes: {unreachable.tolist()}"
        )

    A_root = np.zeros((V, V), dtype=np.float32)
    A_centripetal = np.zeros((V, V), dtype=np.float32)
    A_centrifugal = np.zeros((V, V), dtype=np.float32)

    # self-links always in root
    for i in range(V):
        A_root[i, i] = 1.0

    # partition neighbor edges
    for i, j in neighbor_links:
        di, dj = hop[i], hop[j]
        if dj == di:
            A_root[i, j] = 1.0
            A_root[j, i] = 1.0
        elif dj < di:
            A_centripetal[i, j] = 1.0
            A_centripetal[j, i] = 1.0
        else:
            A_centrifugal[i, j] = 1.0
            A_centrifugal[j, i] = 1.0

    A3 = np.stack(
        [
            normalize_adjacency(A_root),
            normalize_adjacency(A_centripetal),
            normalize_adjacency(A_centrifugal),
        ],
        axis=0,
    )  # (3,V,V)

    return torch.tensor(A3, dtype=torch.float32)


# --------------------------
# Plot helpers
# --------------------------

def save_plot(fig, out_path: str):
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mean_visibility(X: np.ndarray, out_path: str):
    # X: (C,T,V,M)
    vis = X[3, :, :, 0]     # (T,V)
    mean_vis = vis.mean(axis=1)

    fig = plt.figure()
    plt.plot(mean_vis)
    plt.title("Mean visibility over time")
    plt.xlabel("t")
    plt.ylabel("mean visibility")
    save_plot(fig, out_path)


def plot_joint_xy_over_time(X: np.ndarray, joint: int, out_path: str):
    x = X[0, :, joint, 0]
    y = X[1, :, joint, 0]

    fig = plt.figure()
    plt.plot(x, label="x")
    plt.plot(y, label="y")
    plt.title(f"Joint {joint} (x,y) over time")
    plt.xlabel("t")
    plt.legend()
    save_plot(fig, out_path)


def plot_joint_trajectory_xy(X: np.ndarray, joint: int, out_path: str):
    x = X[0, :, joint, 0]
    y = X[1, :, joint, 0]

    fig = plt.figure()
    plt.plot(x, y)
    plt.title(f"Joint {joint} trajectory (x vs y)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().invert_yaxis()
    save_plot(fig, out_path)


def plot_mean_3d_motion(X: np.ndarray, out_path: str):
    coords = X[0:3, :, :, 0]         # (3,T,V)
    d = np.diff(coords, axis=1)      # (3,T-1,V)
    speed = np.linalg.norm(d, axis=0).mean(axis=1)  # (T-1,)

    fig = plt.figure()
    plt.plot(speed)
    plt.title("Mean 3D motion magnitude (frame-to-frame)")
    plt.xlabel("t")
    plt.ylabel("mean ||Δxyz||")
    save_plot(fig, out_path)


def plot_adjacency(A: torch.Tensor, out_dir: str, prefix: str):
    """
    A: (K,V,V)
    Saves one image per partition.
    """
    if out_dir is None:
        return
    A_np = A.detach().cpu().numpy()
    for k in range(A_np.shape[0]):
        fig = plt.figure()
        plt.imshow(A_np[k])
        plt.title(f"{prefix} Adjacency K={k}")
        plt.colorbar()
        out_path = os.path.join(out_dir, f"{prefix.lower()}_adjacency_k{k}.png")
        save_plot(fig, out_path)


# --------------------------
# Main pipeline
# --------------------------

def run_pipeline(
    video_path: str,
    out_dir: str,
    t_target: int,
    debug_ts: list[int],
    joint: int,
    save_visuals: bool = True,
):
    ensure_dir(out_dir)
    frames_dir = os.path.join(out_dir, "frames") if save_visuals else None
    if frames_dir:
        ensure_dir(frames_dir)

    annotated_video_path = os.path.join(out_dir, "annotated.mp4") if save_visuals else None

    # MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    fps, w, h, n_frames = read_video_info(video_path)
    print("Video info:", {"fps": fps, "w": w, "h": h, "n_frames": n_frames})

    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    if save_visuals:
        writer = cv2.VideoWriter(
            annotated_video_path,
            fourcc,
            fps if fps > 0 else 30,
            (w, h),
        )

    raw_frames_bgr: list[np.ndarray] = []
    raw_pose_tensor: list[np.ndarray] = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if save_visuals:
                raw_frames_bgr.append(frame_bgr)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            annotated = frame_bgr.copy()

            if results.pose_landmarks is not None:
                mp_drawing.draw_landmarks(
                    annotated,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )
                raw_pose_tensor.append(landmarks_to_tensor(results.pose_landmarks))
            else:
                raw_pose_tensor.append(empty_tensor())

            if writer is not None:
                writer.write(annotated)
            idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    raw_pose_tensor = np.stack(raw_pose_tensor, axis=0)  # (T_raw, C, V)
    print("raw_pose_tensor:", raw_pose_tensor.shape)
    if save_visuals and annotated_video_path:
        print("Annotated video saved to:", annotated_video_path)

    # Resample to fixed length
    T_raw = raw_pose_tensor.shape[0]
    sample_idx = uniform_sample_indices(T_raw, t_target)

    pose_T = raw_pose_tensor[sample_idx]         # (T, C, V)
    pose_T = np.transpose(pose_T, (1, 0, 2))     # (C, T, V)
    X = pose_T[..., None]                        # (C, T, V, M=1)
    print("X shape:", X.shape, "(C,T,V,M)")

    # Save debug frames + print slices
    if save_visuals and frames_dir:
        with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose_static:
            for t in debug_ts:
                if t < 0 or t >= t_target:
                    continue
                src_idx = int(sample_idx[t])
                frame_bgr = raw_frames_bgr[src_idx].copy()
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                results = pose_static.process(frame_rgb)
                annotated = frame_bgr.copy()
                if results.pose_landmarks is not None:
                    mp_drawing.draw_landmarks(
                        annotated,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )

                out_path = os.path.join(frames_dir, f"frame_t{t:03d}_src{src_idx:05d}.png")
                cv2.imwrite(out_path, annotated)

                tensor_cv = X[:, t, :, 0]  # (C,V)
                print(f"\n--- t={t} (src frame {src_idx}) tensor (C,V) ---")
                print("x[0][:5] =", tensor_cv[0, :5])
                print("y[1][:5] =", tensor_cv[1, :5])
                print("z[2][:5] =", tensor_cv[2, :5])
                print("vis[3][:5] =", tensor_cv[3, :5])

        print("Saved debug frames to:", frames_dir)

    # Plots (full 33-joint)
    plots_dir = os.path.join(out_dir, "plots") if save_visuals else None
    if plots_dir:
        ensure_dir(plots_dir)

        plot_mean_visibility(X, os.path.join(plots_dir, "mean_visibility.png"))
        plot_joint_xy_over_time(X, joint, os.path.join(plots_dir, f"joint{joint}_xy_over_time.png"))
        plot_joint_trajectory_xy(X, joint, os.path.join(plots_dir, f"joint{joint}_trajectory_xy.png"))
        plot_mean_3d_motion(X, os.path.join(plots_dir, "mean_3d_motion.png"))
        print("Saved plots to:", plots_dir)

    # --------------------------
    # Body-only joints (11..32) -> 22 joints
    # --------------------------
    KEEP = list(range(11, 33))  # 22 joints
    old2new = {old_i: new_i for new_i, old_i in enumerate(KEEP)}
    V_BODY = len(KEEP)

    X_body = X[:, :, KEEP, :]  # (4,T,22,1)
    print("X_body:", X_body.shape)

    neighbor_links_full = list(mp.solutions.pose.POSE_CONNECTIONS)
    neighbor_links_body = [
        (old2new[i], old2new[j])
        for (i, j) in neighbor_links_full
        if i in old2new and j in old2new
    ]

    center_body = old2new[23]  # LEFT_HIP remapped
    print("V_BODY:", V_BODY, "edges_body:", len(neighbor_links_body), "center_body:", center_body)

    # Build partitioned adjacency for body graph
    A_body = build_partitioned_adjacency(V_BODY, neighbor_links_body, center=center_body)  # (3,22,22)
    print("A_body:", tuple(A_body.shape), "nonzero:", int((A_body != 0).sum().item()))

    # Plot adjacency partitions
    plot_adjacency(A_body, plots_dir, prefix="BODY")
    print("Saved adjacency plots to:", plots_dir)

    # Torch-ready tensors
    x_in = torch.tensor(X_body, dtype=torch.float32).unsqueeze(0)  # (N=1,C,T,V,M)
    print("x_in:", tuple(x_in.shape), "(N,C,T,V,M)")
    print("A_body:", tuple(A_body.shape), "(K,V,V)")

    # Save outputs
    npz_path = os.path.join(out_dir, "pose_outputs.npz")
    pt_path = os.path.join(out_dir, "pose_outputs.pt")

    np.savez_compressed(
        npz_path,
        X=X,
        X_body=X_body,
        sample_idx=sample_idx,
        KEEP=np.array(KEEP, dtype=np.int32),
        neighbor_links_body=np.array(neighbor_links_body, dtype=np.int32),
        center_body=np.int32(center_body),
        A_body=A_body.detach().cpu().numpy(),
    )

    torch.save(
        {
            "x_in": x_in,
            "A_body": A_body,
            "sample_idx": torch.tensor(sample_idx, dtype=torch.long),
            "KEEP": torch.tensor(KEEP, dtype=torch.long),
            "neighbor_links_body": torch.tensor(neighbor_links_body, dtype=torch.long),
            "center_body": torch.tensor(center_body, dtype=torch.long),
        },
        pt_path,
    )

    print("Saved tensors to:")
    print(" -", npz_path)
    print(" -", pt_path)
    print("Done.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="Path to input video")
    p.add_argument("--out_dir", default="./debug_pose", help="Output directory")
    p.add_argument("--t_target", type=int, default=60, help="Resampled clip length T")
    p.add_argument("--debug_ts", type=int, nargs="*", default=[0, 10, 20, 30, 40, 50],
                   help="Debug timesteps in resampled timeline (0..T-1)")
    p.add_argument("--joint", type=int, default=24, help="Joint index (full 33-joint indexing) for plots")
    p.add_argument("--no_visuals", dest="save_visuals", action="store_false",
                   help="Skip annotated video / frames / plots to speed up dataset preparation")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        video_path=args.video,
        out_dir=args.out_dir,
        t_target=args.t_target,
        debug_ts=args.debug_ts,
        joint=args.joint,
        save_visuals=args.save_visuals,
    )
