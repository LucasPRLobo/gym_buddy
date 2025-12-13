#!/usr/bin/env python3
"""
Sanity-training loop for ST-GCN.

Goal:
- Load x_in and A_body from pose_outputs.pt
- Create a dummy label (single class)
- Train for a small number of steps
- Verify: loss decreases + gradients flow
- Save: model checkpoint + (optional) loss plot

Usage:
  python train_sanity.py \
    --pt ./debug_pose/pose_outputs.pt \
    --num_classes 10 \
    --label 0 \
    --steps 200 \
    --lr 1e-3 \
    --device cpu \
    --out_dir ./debug_pose/sanity
"""

import os
import argparse
import random

import numpy as np
import torch
import torch.nn as nn

# Import your model definition (must expose STGCN or similar).
# Adjust if your class name/module differs.
from stgcn_model import STGCN


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pose_pt(pt_path: str, device: torch.device):
    obj = torch.load(pt_path, map_location="cpu")

    if isinstance(obj, dict):
        if "x_in" not in obj or "A_body" not in obj:
            raise KeyError(f"{pt_path} must contain keys 'x_in' and 'A_body'. Found: {list(obj.keys())}")
        x_in = obj["x_in"]
        A_body = obj["A_body"]
    else:
        raise TypeError(f"{pt_path} must be a dict saved by torch.save({...}). Got: {type(obj)}")

    # Keep A on CPU or move to device (both fine); for simplicity, move both.
    x_in = x_in.to(device).float()
    A_body = A_body.to(device).float()

    # Expected:
    # x_in: (N,C,T,V,M)
    # A_body: (K,V,V)
    if x_in.ndim != 5:
        raise ValueError(f"x_in should be 5D (N,C,T,V,M). Got shape {tuple(x_in.shape)}")
    if A_body.ndim != 3:
        raise ValueError(f"A_body should be 3D (K,V,V). Got shape {tuple(A_body.shape)}")

    return x_in, A_body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="Path to pose_outputs.pt (must contain x_in and A_body).")
    ap.add_argument("--num_classes", type=int, default=10, help="Number of classes for the classifier head.")
    ap.add_argument("--label", type=int, default=0, help="Dummy target label (0..num_classes-1).")
    ap.add_argument("--steps", type=int, default=200, help="Optimization steps.")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    ap.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--out_dir", default="./sanity_out", help="Where to save checkpoint/plot.")
    ap.add_argument("--plot", action="store_true", help="If set, saves a loss curve PNG (requires matplotlib).")
    ap.add_argument("--save_every", type=int, default=0, help="If >0, saves intermediate checkpoints every N steps.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    set_seed(args.seed)

    # 1) Load one clip tensor + adjacency
    x_in, A_body = load_pose_pt(args.pt, device=device)
    N = x_in.shape[0]
    if N != 1:
        print(f"[warn] sanity script expects N=1 typically, but got N={N}. Proceeding anyway.")

    if not (0 <= args.label < args.num_classes):
        raise ValueError(f"--label must be in [0, {args.num_classes-1}]")

    # 2) Build model
    model = STGCN(num_classes=args.num_classes, in_channels=x_in.shape[1])  # in_channels=4 typically
    model = model.to(device)
    model.train()

    # 3) Dummy target
    y = torch.full((x_in.shape[0],), args.label, dtype=torch.long, device=device)

    # 4) Optim + loss
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # 5) Train loop
    losses = []
    print("Loaded:")
    print(" x_in:", tuple(x_in.shape))
    print(" A_body:", tuple(A_body.shape))
    print("Device:", device)
    print(f"Target label: {args.label} / num_classes={args.num_classes}")
    print("Training...")

    for step in range(1, args.steps + 1):
        optim.zero_grad(set_to_none=True)

        logits = model(x_in, A_body)  # expected (N, num_classes)
        loss = criterion(logits, y)

        loss.backward()
        optim.step()

        losses.append(float(loss.item()))

        if step == 1 or step % 10 == 0 or step == args.steps:
            with torch.no_grad():
                probs = torch.softmax(logits, dim=1)
                pred = int(torch.argmax(probs, dim=1).item())
                p_target = float(probs[0, args.label].item())
            print(f"step {step:4d}/{args.steps} | loss={loss.item():.6f} | pred={pred} | p(target)={p_target:.4f}")

        if args.save_every and (step % args.save_every == 0):
            ckpt_path = os.path.join(args.out_dir, f"stgcn_sanity_step{step}.pt")
            torch.save(
                {
                    "step": step,
                    "model_state": model.state_dict(),
                    "optim_state": optim.state_dict(),
                    "loss": losses[-1],
                    "args": vars(args),
                },
                ckpt_path,
            )

    # 6) Save final checkpoint
    final_path = os.path.join(args.out_dir, "stgcn_sanity_final.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
            "losses": losses,
            "args": vars(args),
        },
        final_path,
    )

    # 7) Optional plot
    if args.plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(losses)
        plt.title("Sanity training loss")
        plt.xlabel("step")
        plt.ylabel("cross-entropy loss")
        plot_path = os.path.join(args.out_dir, "loss_curve.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print("Saved loss plot to:", plot_path)

    print("Saved final checkpoint to:", final_path)
    print("Done.")


if __name__ == "__main__":
    main()
