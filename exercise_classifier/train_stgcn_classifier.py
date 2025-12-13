#!/usr/bin/env python3
"""
Train/test the ST-GCN classifier on precomputed pose tensors.

Workflow:
 1) Convert videos to pose tensors (one folder per video):
      python prepare_pose_dataset.py \
        --videos_root datasets/workout_fitness_hasyim \
        --out_root outputs/pose_hasyim \
        --max_per_class 5

 2) Train and evaluate:
      python train_stgcn_classifier.py \
        --data_root outputs/pose_hasyim \
        --epochs 20 \
        --batch_size 8 \
        --device cuda

The script expects each sample directory to contain pose_outputs.pt with keys
`x_in` (1,C,T,V,M) and `A_body` (K,V,V).
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler

from stgcn_model import STGCN


@dataclass
class Sample:
    pt_path: Path
    label: str


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _temporal_fit(x: torch.Tensor, target_T: int, jitter: int = 0) -> torch.Tensor:
    # x: (C,T,V,M)
    C, T, V, M = x.shape
    if T <= 0:
        raise ValueError("Temporal length T must be > 0")

    # Build indices with optional jitter around uniform sampling.
    idx = torch.linspace(0, T - 1, target_T)
    if jitter > 0:
        noise = torch.randint(-jitter, jitter + 1, (target_T,), dtype=torch.long)
        idx = idx + noise
    idx = idx.round().long().clamp(0, T - 1)
    return x[:, idx, :, :]


class PoseClipDataset(Dataset):
    def __init__(self, samples: List[Sample], target_T: int | None = None, jitter: int = 0, is_train: bool = False):
        if not samples:
            raise ValueError("PoseClipDataset requires at least one sample")
        self.samples = samples
        self.target_T = target_T
        self.jitter = jitter if is_train else 0

        probe = torch.load(samples[0].pt_path, map_location="cpu")
        if "x_in" not in probe or "A_body" not in probe:
            raise KeyError(f"{samples[0].pt_path} missing x_in/A_body keys")

        x0 = probe["x_in"]
        self.A_body = probe["A_body"].float()

        if x0.dim() == 5 and x0.shape[0] == 1:
            x0 = x0[0]
        if x0.dim() != 4:
            raise ValueError(f"x_in should be (C,T,V,M) or (1,C,T,V,M), got {tuple(x0.shape)}")

        self.C, self.T, self.V, self.M = x0.shape
        if self.M != 1:
            raise ValueError(f"Only single-person clips supported (M=1). Got M={self.M}")

        if target_T is None:
            self.target_T = self.T

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        obj = torch.load(sample.pt_path, map_location="cpu")
        x = obj["x_in"]
        if x.dim() == 5 and x.shape[0] == 1:
            x = x[0]
        x = x.float()

        if x.shape[1] != self.target_T or self.jitter > 0:
            x = _temporal_fit(x, self.target_T, jitter=self.jitter)

        return x, sample.label


def load_manifest(data_root: Path) -> List[Sample]:
    manifest_path = data_root / "dataset_index.json"
    samples: List[Sample] = []

    if manifest_path.exists():
        data = json.loads(manifest_path.read_text())
        for row in data:
            samples.append(Sample(pt_path=Path(row["pt"]), label=row["label"]))
        return samples

    # Fallback: scan for pose_outputs.pt under <data_root>/<label>/<video>/
    for pt_path in sorted(data_root.rglob("pose_outputs.pt")):
        parts = pt_path.parts
        # Expect .../<label>/<video>/pose_outputs.pt
        if len(parts) < 2:
            continue
        # derive label as immediate parent of video folder if present
        label = pt_path.parent.parent.name.lower()
        samples.append(Sample(pt_path=pt_path, label=label))

    return samples


def encode_labels(samples: List[Sample]):
    labels = sorted({s.label for s in samples})
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    return label_to_idx, labels


def collate_batch(batch, label_to_idx):
    xs, labels = zip(*batch)
    x_tensor = torch.stack(xs, dim=0)  # (N,C,T,V,M)
    y = torch.tensor([label_to_idx[lbl] for lbl in labels], dtype=torch.long)
    return x_tensor, y


def train_one_epoch(model, loader, A_body, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x, A_body)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == y).sum().item())
        total += x.size(0)

    return total_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def evaluate(model, loader, A_body, criterion, device, return_preds: bool = False):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    all_preds = []
    all_targets = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x, A_body)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == y).sum().item())
        total += x.size(0)
        if return_preds:
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())
    avg_loss = total_loss / max(1, total)
    avg_acc = correct / max(1, total)
    if return_preds:
        return avg_loss, avg_acc, torch.cat(all_preds), torch.cat(all_targets)
    return avg_loss, avg_acc


def split_samples(samples: List[Sample], val_split: float, test_split: float, seed: int):
    total = len(samples)
    n_test = int(total * test_split)
    n_val = int((total - n_test) * val_split)
    n_train = total - n_test - n_val
    generator = torch.Generator().manual_seed(seed)
    return random_split(samples, [n_train, n_val, n_test], generator=generator)


def class_counts(samples: List[Sample]):
    counts = {}
    for s in samples:
        counts[s.label] = counts.get(s.label, 0) + 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Root with processed pose outputs")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--test_split", type=float, default=0.15)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target_T", type=int, default=None, help="Force all clips to this T (default: infer)")
    ap.add_argument("--out_dir", default="./stgcn_runs", help="Where to save checkpoints/metrics")
    ap.add_argument("--eval_only", action="store_true", help="Only run evaluation on --ckpt")
    ap.add_argument("--ckpt", default=None, help="Checkpoint path for eval_only or warm start")
    ap.add_argument("--use_class_weights", action="store_true", help="Reweight loss by inverse class frequency")
    ap.add_argument("--balanced_sampler", action="store_true", help="Use weighted sampling to balance batches")
    ap.add_argument("--dropout", type=float, default=0.35, help="Dropout used in ST-GCN blocks")
    ap.add_argument("--jitter", type=int, default=2, help="Temporal jitter (+/- frames) for training augmentation")
    ap.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs without val gain)")
    ap.add_argument("--lr_scheduler", choices=["none", "step", "cosine", "plateau"], default="none")
    ap.add_argument("--step_size", type=int, default=5, help="StepLR step size (if lr_scheduler=step)")
    ap.add_argument("--gamma", type=float, default=0.5, help="LR decay factor for StepLR")
    ap.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing for CE loss")
    ap.add_argument("--report_metrics", action="store_true", help="Print confusion matrix and per-class report on test")
    args = ap.parse_args()

    set_seed(args.seed)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_manifest(data_root)
    if not samples:
        raise RuntimeError(f"No pose_outputs.pt files found under {data_root}")

    label_to_idx, labels = encode_labels(samples)
    print(f"[INFO] Classes ({len(labels)}): {labels}")
    print(f"[INFO] Total samples: {len(samples)}")

    train_s, val_s, test_s = split_samples(samples, args.val_split, args.test_split, args.seed)
    print(f"[SPLIT] train={len(train_s)}, val={len(val_s)}, test={len(test_s)}")

    train_ds = PoseClipDataset(list(train_s), target_T=args.target_T, jitter=args.jitter, is_train=True)
    val_ds = PoseClipDataset(list(val_s), target_T=args.target_T)
    test_ds = PoseClipDataset(list(test_s), target_T=args.target_T)

    # Verify adjacency consistency
    A_body = train_ds.A_body
    for ds, name in [(val_ds, "val"), (test_ds, "test")]:
        if not torch.equal(A_body, ds.A_body):
            raise ValueError(f"A_body mismatch between train and {name} splits")

    A_body = A_body.to(args.device)

    def collate_fn(batch):
        return collate_batch(batch, label_to_idx)

    train_sampler = None
    if args.balanced_sampler:
        counts = class_counts(train_s)
        print(f"[BALANCE] train class counts: {counts}")
        weights = [1.0 / counts[s.label] for s in train_s]
        train_sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = STGCN(
        in_channels=train_ds.C,
        num_classes=len(labels),
        V=train_ds.V,
        K=A_body.shape[0],
        dropout=args.dropout,
    ).to(args.device)

    if args.use_class_weights:
        counts = class_counts(train_s)
        weights = torch.tensor(
            [1.0 / counts[lbl] for lbl in labels],
            dtype=torch.float32,
            device=args.device,
        )
        print(f"[BALANCE] loss class weights: {weights.tolist()}")
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    start_epoch = 0
    best_val_acc = -1.0
    best_state = None
    epochs_no_improve = 0

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt and not args.eval_only:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"[CKPT] Loaded from {args.ckpt} (epoch {start_epoch})")

    if args.eval_only:
        val_loss, val_acc = evaluate(model, val_loader, A_body, criterion, args.device)
        if args.report_metrics:
            test_loss, test_acc, preds, targets = evaluate(
                model, test_loader, A_body, criterion, args.device, return_preds=True
            )
            print(f"[EVAL] val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
            print(f"[EVAL] test_loss={test_loss:.4f} test_acc={test_acc:.4f}")
            print("[TEST] confusion matrix:\n", confusion_matrix(targets, preds))
            print("[TEST] classification report:\n", classification_report(targets, preds, target_names=labels, digits=3))
        else:
            test_loss, test_acc = evaluate(model, test_loader, A_body, criterion, args.device)
            print(f"[EVAL] val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
            print(f"[EVAL] test_loss={test_loss:.4f} test_acc={test_acc:.4f}")
        return

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, A_body, criterion, optimizer, args.device)
        val_loss, val_acc = evaluate(model, val_loader, A_body, criterion, args.device)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        print(
            f"Epoch {epoch+1:03d}/{args.epochs} "
            f"| train_loss={train_loss:.4f} acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f} acc={val_acc:.4f}"
        )

        is_best = val_acc > best_val_acc
        best_val_acc = max(best_val_acc, val_acc)
        if is_best:
            epochs_no_improve = 0
            best_state = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "labels": labels,
                "label_to_idx": label_to_idx,
                "best_val_acc": best_val_acc,
                "args": vars(args),
            }
        else:
            epochs_no_improve += 1

        ckpt_obj = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "labels": labels,
            "label_to_idx": label_to_idx,
            "best_val_acc": best_val_acc,
            "args": vars(args),
        }
        torch.save(ckpt_obj, out_dir / "last.pt")
        if is_best:
            torch.save(ckpt_obj, out_dir / "best.pt")

        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f"[EARLY STOP] No val improvement for {args.patience} epochs.")
            break

    if best_state is not None:
        model.load_state_dict(best_state["model_state"])
        print(f"[LOAD BEST] epoch={best_state['epoch']} val_acc={best_state['best_val_acc']:.4f}")

    if args.report_metrics:
        test_loss, test_acc, preds, targets = evaluate(
            model, test_loader, A_body, criterion, args.device, return_preds=True
        )
        all_labels = list(range(len(labels)))
        print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f}")
        print("[TEST] confusion matrix:\n", confusion_matrix(targets, preds, labels=all_labels))
        print("[TEST] classification report:\n", classification_report(targets, preds, labels=all_labels, target_names=labels, digits=3, zero_division=0))
    else:
        test_loss, test_acc = evaluate(model, test_loader, A_body, criterion, args.device)
        print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f}")

    print(f"[DONE] Best val acc={best_val_acc:.4f}. Checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
