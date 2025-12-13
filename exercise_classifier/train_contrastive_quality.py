#!/usr/bin/env python3
"""
Train a contrastive ST-GCN encoder to produce robust motion embeddings.

Uses supervised contrastive loss across exercise labels on precomputed pose tensors.
Embeddings can be plugged into quality_eval.py via --ckpt for more stable quality scoring.

Example:
  python exercise_classifier/train_contrastive_quality.py \
    --data_root outputs/pose_hasyim \
    --epochs 30 \
    --batch_size 16 \
    --device cuda \
    --temperature 0.1 \
    --dropout 0.35 \
    --jitter 3 \
    --out_dir outputs/stgcn_contrastive
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from stgcn_model import STGCN


class Sample:
    def __init__(self, pt_path: Path, label: str):
        self.pt_path = pt_path
        self.label = label


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _temporal_fit(x: torch.Tensor, target_T: int, jitter: int = 0) -> torch.Tensor:
    # x: (C,T,V,M)
    C, T, V, M = x.shape
    idx = torch.linspace(0, T - 1, target_T)
    if jitter > 0:
        noise = torch.randint(-jitter, jitter + 1, (target_T,), dtype=torch.long)
        idx = idx + noise
    idx = idx.round().long().clamp(0, T - 1)
    return x[:, idx, :, :]


class PoseClipDataset(Dataset):
    def __init__(self, samples: List[Sample], target_T: int | None, jitter: int, is_train: bool):
        if not samples:
            raise ValueError("PoseClipDataset requires at least one sample")
        self.samples = samples
        self.target_T = target_T
        self.jitter = jitter if is_train else 0

        probe = torch.load(samples[0].pt_path, map_location="cpu")
        x0 = probe["x_in"]
        if x0.dim() == 5 and x0.shape[0] == 1:
            x0 = x0[0]
        self.C, self.T, self.V, self.M = x0.shape
        self.A_body = probe["A_body"].float()
        if self.target_T is None:
            self.target_T = self.T

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        obj = torch.load(s.pt_path, map_location="cpu")
        x = obj["x_in"]
        if x.dim() == 5 and x.shape[0] == 1:
            x = x[0]
        x = x.float()
        if x.shape[1] != self.target_T or self.jitter > 0:
            x = _temporal_fit(x, self.target_T, jitter=self.jitter)
        return x, s.label


def load_manifest(data_root: Path) -> List[Sample]:
    manifest_path = data_root / "dataset_index.json"
    samples: List[Sample] = []
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text())
        for row in data:
            samples.append(Sample(pt_path=Path(row["pt"]), label=row["label"]))
    else:
        for pt in sorted(data_root.rglob("pose_outputs.pt")):
            if len(pt.parts) < 2:
                continue
            label = pt.parent.parent.name.lower()
            samples.append(Sample(pt_path=pt, label=label))
    return samples


def encode_labels(samples: List[Sample]):
    labels = sorted({s.label for s in samples})
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    return label_to_idx, labels


def supervised_contrastive_loss(emb: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1):
    """
    emb: (N,D) normalized embeddings
    labels: (N,)
    """
    device = emb.device
    emb = nn.functional.normalize(emb, dim=1)
    sim = torch.matmul(emb, emb.T) / temperature  # (N,N)
    # mask self
    logits_mask = torch.ones_like(sim) - torch.eye(sim.size(0), device=device)
    sim = sim - sim.max(dim=1, keepdim=True).values  # stability
    exp_logits = torch.exp(sim) * logits_mask

    mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    mask = mask * logits_mask
    denom = exp_logits.sum(dim=1, keepdim=True) + 1e-8
    log_prob = sim - torch.log(denom)

    pos_count = mask.sum(dim=1)
    mean_log_prob_pos = torch.where(
        pos_count > 0,
        (mask * log_prob).sum(dim=1) / (pos_count + 1e-8),
        torch.zeros_like(pos_count),
    )
    valid = pos_count > 0
    if valid.sum() == 0:
        return None
    loss = -mean_log_prob_pos[valid].mean()
    return loss


def collate_batch(batch, label_to_idx):
    xs, labels = zip(*batch)
    x_tensor = torch.stack(xs, dim=0)  # (N,C,T,V,M)
    y = torch.tensor([label_to_idx[lbl] for lbl in labels], dtype=torch.long)
    return x_tensor, y


@torch.no_grad()
def evaluate_centroid(model, loader, A_body, labels: List[str], device):
    model.eval()
    embs = []
    ys = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        e = model(x, A_body, return_embedding=True)
        embs.append(nn.functional.normalize(e, dim=1))
        ys.append(y)
    embs = torch.cat(embs, dim=0)
    ys = torch.cat(ys, dim=0)
    centroids = []
    for i in range(len(labels)):
        mask = ys == i
        if mask.sum() == 0:
            centroids.append(torch.zeros_like(embs[0]))
        else:
            centroids.append(embs[mask].mean(dim=0))
    centroids = torch.stack(centroids, dim=0)  # (C,D)
    sim = torch.matmul(embs, centroids.T)  # (N,C)
    preds = torch.argmax(sim, dim=1)
    acc = (preds == ys).float().mean().item()
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Root with processed pose tensors")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target_T", type=int, default=None)
    ap.add_argument("--jitter", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.35)
    ap.add_argument("--out_dir", default="./stgcn_contrastive")
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_manifest(Path(args.data_root))
    if not samples:
        raise RuntimeError("No samples found for contrastive training.")

    label_to_idx, labels = encode_labels(samples)
    print(f"[INFO] Classes ({len(labels)}): {labels}")
    generator = torch.Generator().manual_seed(args.seed)
    n_val = int(len(samples) * args.val_split)
    n_train = len(samples) - n_val
    train_s, val_s = random_split(samples, [n_train, n_val], generator=generator)

    train_ds = PoseClipDataset(list(train_s), target_T=args.target_T, jitter=args.jitter, is_train=True)
    val_ds = PoseClipDataset(list(val_s), target_T=args.target_T, jitter=args.jitter, is_train=False)

    if not torch.equal(train_ds.A_body, val_ds.A_body):
        raise ValueError("A_body mismatch between train/val splits.")
    A_body = train_ds.A_body.to(args.device)

    def collate_fn(batch):
        return collate_batch(batch, label_to_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = STGCN(
        in_channels=train_ds.C,
        num_classes=len(labels),
        V=train_ds.V,
        K=A_body.shape[0],
        dropout=args.dropout,
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batch = 0
        for x, y in train_loader:
            x = x.to(args.device)
            y = y.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            emb = model(x, A_body, return_embedding=True)
            loss = supervised_contrastive_loss(emb, y, temperature=args.temperature)
            if loss is None or not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.item())
            n_batch += 1
        train_loss = total_loss / max(1, n_batch)

        val_acc = evaluate_centroid(model, val_loader, A_body, labels, args.device)
        print(f"Epoch {epoch+1:03d}/{args.epochs} | train_contrastive_loss={train_loss:.4f} | val_centroid_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            ckpt = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "labels": labels,
                "label_to_idx": label_to_idx,
                "best_val_acc": best_val,
                "args": {
                    "V": train_ds.V,
                    "K": A_body.shape[0],
                    "in_channels": train_ds.C,
                    "dropout": args.dropout,
                    "target_T": train_ds.target_T,
                },
            }
            torch.save(ckpt, out_dir / "best.pt")
            torch.save(ckpt, out_dir / "last.pt")

    print(f"[DONE] Best val centroid acc={best_val:.4f}. Checkpoints saved to {out_dir}")


if __name__ == "__main__":
    main()
