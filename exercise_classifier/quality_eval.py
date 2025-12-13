#!/usr/bin/env python3
"""
Quality scoring prototype:
- Reuse a trained ST-GCN classifier backbone as an encoder.
- Compare a user clip embedding to a set of reference embeddings (same exercise).
- Emit a simple quality score + basic feedback hints.

Usage example:
  python quality_eval.py \
    --clip outputs/pose_hasyim/squat/squat_01/pose_outputs.pt \
    --ref_root outputs/pose_hasyim \
    --label squat \
    --ckpt outputs/stgcn_hasyim_sampler_only/best.pt \
    --device cuda
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from stgcn_model import STGCN

# 22 body joints mapping (MediaPipe 11..32)
BODY_JOINT_NAMES = [
    "left_shoulder",   # 0 -> 11
    "right_shoulder",  # 1 -> 12
    "left_elbow",      # 2 -> 13
    "right_elbow",     # 3 -> 14
    "left_wrist",      # 4 -> 15
    "right_wrist",     # 5 -> 16
    "left_pinky",      # 6 -> 17
    "right_pinky",     # 7 -> 18
    "left_index",      # 8 -> 19
    "right_index",     # 9 -> 20
    "left_thumb",      # 10 -> 21
    "right_thumb",     # 11 -> 22
    "left_hip",        # 12 -> 23
    "right_hip",       # 13 -> 24
    "left_knee",       # 14 -> 25
    "right_knee",      # 15 -> 26
    "left_ankle",      # 16 -> 27
    "right_ankle",     # 17 -> 28
    "left_heel",       # 18 -> 29
    "right_heel",      # 19 -> 30
    "left_foot_index", # 20 -> 31
    "right_foot_index" # 21 -> 32
]


def load_clip(pt_path: Path, device: torch.device):
    obj = torch.load(pt_path, map_location="cpu")
    if "x_in" not in obj or "A_body" not in obj:
        raise KeyError(f"{pt_path} missing x_in or A_body")
    x = obj["x_in"]
    if x.dim() == 5 and x.shape[0] == 1:
        x = x[0]
    A = obj["A_body"]
    return x.to(device).float().unsqueeze(0), A.to(device).float()


def load_refs(ref_root: Path, label: str, max_refs: int, device: torch.device):
    manifest_path = ref_root / "dataset_index.json"
    refs: List[Path] = []
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text())
        for row in data:
            if row["label"] == label:
                refs.append(Path(row["pt"]))
    else:
        for pt in ref_root.rglob("pose_outputs.pt"):
            if pt.parts and label == pt.parent.parent.name.lower():
                refs.append(pt)

    if not refs:
        raise RuntimeError(f"No reference clips found for label '{label}' under {ref_root}")
    random.shuffle(refs)
    refs = refs[:max_refs]
    clips = []
    for pt_path in refs:
        clips.append(load_clip(pt_path, device))
    return clips


def build_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt.get("args", {})
    V = args.get("V", 22)
    K = args.get("K", 3)
    in_channels = args.get("in_channels", 4)
    num_classes = len(ckpt.get("labels", [])) or args.get("num_classes", 10)
    dropout = args.get("dropout", 0.35)
    model = STGCN(in_channels=in_channels, num_classes=num_classes, V=V, K=K, dropout=dropout)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def embedding_from_clip(model: STGCN, x: torch.Tensor, A: torch.Tensor):
    with torch.no_grad():
        emb = model(x, A, return_embedding=True)  # (1,256)
    return F.normalize(emb, dim=1)


def score_quality(user_emb: torch.Tensor, ref_embs: torch.Tensor):
    # Cosine similarity
    cos = F.cosine_similarity(user_emb, ref_embs).mean().item()
    # L2 distance to mean ref
    ref_mean = ref_embs.mean(dim=0, keepdim=True)
    l2 = torch.norm(user_emb - ref_mean, p=2).item()
    # Map to 0..100 (simple heuristic: higher cosine, lower l2)
    cos_score = max(0.0, min(1.0, (cos + 1) / 2))
    l2_score = 1.0 / (1.0 + l2)
    quality = 100.0 * (0.6 * cos_score + 0.4 * l2_score)
    return quality, cos, l2


def motion_stats(x: torch.Tensor):
    # x: (1,C,T,V,M) -> use xyz only, compute mean frame-to-frame motion as a performance proxy
    coords = x[0, 0:3]  # (3,T,V,M)
    coords = coords.permute(1, 2, 3, 0)  # (T,V,M,3)
    diff = coords[1:] - coords[:-1]
    speed = diff.norm(dim=-1).mean().item()
    return speed


def feedback_messages(quality: float, cos: float, l2: float, user_speed: float, ref_speed: float):
    form_msgs = []
    perf_msgs = []

    if quality < 50:
        form_msgs.append("Form is far from reference; focus on stable posture and full range of motion.")
    elif quality < 70:
        form_msgs.append("Form moderately deviates; refine joint tracking and consistency across reps.")
    else:
        form_msgs.append("Form close to reference; keep consistency and control.")

    if cos < 0.6:
        form_msgs.append("Motion pattern diverges from expert baseline; align sequencing of phases.")
    if l2 > 2.0:
        form_msgs.append("Large spatial deviation; check joint alignment and range of motion.")

    if ref_speed > 1e-6:
        ratio = user_speed / ref_speed
        if ratio < 0.7:
            perf_msgs.append("Pace is slower than reference; push a bit harder while keeping control.")
        elif ratio > 1.3:
            perf_msgs.append("Pace is faster than reference; slow slightly to improve control and depth.")
        else:
            perf_msgs.append("Pace matches reference; maintain steady tempo.")

    return form_msgs, perf_msgs


def per_joint_deviation(user_x: torch.Tensor, ref_xs: List[torch.Tensor]):
    """
    Compute per-joint deviation between user and reference mean over time and xyz.
    Returns sorted list of (joint_idx, deviation_value).
    """
    # user_x: (1,C,T,V,M)
    user = user_x[0, 0:3]  # (3,T,V,M)
    user = user.permute(1, 2, 3, 0)  # (T,V,M,3)
    ref_stack = torch.stack([rx[0, 0:3].permute(1, 2, 3, 0) for rx in ref_xs], dim=0)  # (R,T,V,M,3)
    ref_mean = ref_stack.mean(dim=0)  # (T,V,M,3)

    diff = user - ref_mean  # (T,V,M,3)
    # mean over time and person, norm over xyz
    per_joint = diff.norm(dim=-1).mean(dim=(0, 2))  # (V,)
    sorted_idx = torch.argsort(per_joint, descending=True)
    return [(int(idx), float(per_joint[idx].item())) for idx in sorted_idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, help="User clip pose_outputs.pt")
    ap.add_argument("--label", required=True, help="Exercise label (must match reference label)")
    ap.add_argument("--ref_root", required=True, help="Root of reference pose tensors (with dataset_index.json)")
    ap.add_argument("--ckpt", required=True, help="Trained ST-GCN checkpoint to reuse as encoder")
    ap.add_argument("--max_refs", type=int, default=20, help="Max reference clips to compare")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--top_joints", type=int, default=3, help="Number of top deviating joints to report")
    args = ap.parse_args()

    # Select device with fallback if CUDA not available
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model = build_model(Path(args.ckpt), device)
    user_x, user_A = load_clip(Path(args.clip), device)
    refs = load_refs(Path(args.ref_root), args.label, args.max_refs, device)

    user_emb = embedding_from_clip(model, user_x, user_A)
    ref_embs = torch.cat([embedding_from_clip(model, x, A) for x, A in refs], dim=0)

    quality, cos, l2 = score_quality(user_emb, ref_embs)

    user_speed = motion_stats(user_x)
    ref_speed = np.mean([motion_stats(x) for x, _ in refs])

    print("Quality score (0-100):", round(quality, 2))
    print("Cosine similarity:", round(cos, 4))
    print("L2 distance:", round(l2, 4))
    print("User speed (perf proxy):", round(user_speed, 6), "Ref speed:", round(ref_speed, 6))

    form_msgs, perf_msgs = feedback_messages(quality, cos, l2, user_speed, ref_speed)
    print("\nFeedback - Form:")
    for m in form_msgs:
        print(" -", m)
    if perf_msgs:
        print("\nFeedback - Performance:")
        for m in perf_msgs:
            print(" -", m)

    # Per-joint deviations
    top = per_joint_deviation(user_x, [x for x, _ in refs])[: args.top_joints]
    print("\nTop joint deviations:")
    for idx, val in top:
        name = BODY_JOINT_NAMES[idx] if idx < len(BODY_JOINT_NAMES) else f"joint_{idx}"
        hint = "stabilize alignment" if "shoulder" in name or "hip" in name else "control tracking through range"
        print(f" - {name}: deviation={val:.4f} -> {hint}")


if __name__ == "__main__":
    main()
