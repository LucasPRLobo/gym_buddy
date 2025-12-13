#!/usr/bin/env python3
"""
Minimal ST-GCN (PyTorch) that consumes:
  x: (N, C, T, V, M)
  A: (K, V, V)

Implements:
- Data BN across (M*V*C) as in classic ST-GCN
- Spatial graph conv via adjacency partitions K
- Temporal conv (Conv2d over (T,V) with kernel (Kt,1))
- Global average pooling -> logits

Run a sanity test:
  python stgcn_model.py --pt ./debug_pose/pose_outputs.pt --num_classes 10
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


class STGCNBlock(nn.Module):
    """
    One ST-GCN block:
      - Spatial graph convolution using K adjacency partitions
      - Temporal convolution over time
      - Residual connection
    """
    def __init__(self, in_channels, out_channels, K, stride=1, residual=True, dropout=0.0, Kt=9):
        super().__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Spatial graph conv: 1x1 conv produces K*out_channels channels, later split into K groups
        self.gcn = nn.Conv2d(in_channels, out_channels * K, kernel_size=1)

        # Temporal conv: conv over time dimension only (Kt,1)
        pad = (Kt - 1) // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(Kt, 1), stride=(stride, 1), padding=(pad, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )

        # Residual
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        """
        x: (N, Cin, T, V)
        A: (K, V, V)
        """
        N, Cin, T, V = x.shape
        assert A.shape[0] == self.K and A.shape[1] == V and A.shape[2] == V, \
            f"A must be (K,V,V)=({self.K},{V},{V}), got {tuple(A.shape)}"

        # Spatial GCN
        y = self.gcn(x)  # (N, K*Cout, T, V)
        y = y.view(N, self.K, self.out_channels, T, V)  # (N,K,Cout,T,V)

        # Aggregate partitions using adjacency
        # einsum: (N,K,C,T,V) x (K,V,W) -> (N,C,T,W)
        y = torch.einsum('nkctv,kvw->nctw', y, A)

        # Temporal conv + residual
        y = self.tcn(y) + self.residual(x)
        return self.relu(y)


class STGCN(nn.Module):
    """
    Minimal ST-GCN backbone for classification.
    Input x: (N, C, T, V, M)
    """
    def __init__(self, in_channels=4, num_classes=10, V=22, K=3, dropout=0.25):
        super().__init__()
        self.V = V
        self.K = K
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Data BN over (M*V*C)
        # We'll reshape to (N, M*V*C, T) then BN1d.
        self.data_bn = nn.BatchNorm1d(in_channels * V)  # M=1 here; if M>1 adjust at runtime

        # Blocks (small but reasonable for testing)
        self.l1 = STGCNBlock(in_channels, 64, K=K, residual=False, dropout=dropout)
        self.l2 = STGCNBlock(64, 64, K=K, dropout=dropout)
        self.l3 = STGCNBlock(64, 128, K=K, stride=2, dropout=dropout)  # temporal downsample
        self.l4 = STGCNBlock(128, 128, K=K, dropout=dropout)
        self.l5 = STGCNBlock(128, 256, K=K, stride=2, dropout=dropout)  # temporal downsample
        self.l6 = STGCNBlock(256, 256, K=K, dropout=dropout)

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x, A, return_embedding: bool = False):
        """
        x: (N, C, T, V, M)
        A: (K, V, V)
        """
        assert x.dim() == 5, f"Expected x to be 5D (N,C,T,V,M), got {x.shape}"
        N, C, T, V, M = x.shape
        assert V == self.V, f"Model V={self.V} but input V={V}"
        assert A.shape == (self.K, V, V), f"A must be {(self.K, V, V)}, got {tuple(A.shape)}"

        # Merge person dimension into batch for pooling later (classic ST-GCN does N,M flatten)
        # But BN is typically over (N, M*V*C, T)
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # (N,M,V,C,T)
        x = x.view(N, M * V * C, T)                # (N, M*V*C, T)

        # If M != 1, re-init BN would be needed; for now M=1 in your pipeline
        if M != 1:
            raise ValueError(f"This minimal model assumes M=1. Got M={M}.")

        x = self.data_bn(x)                        # (N, V*C, T)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous()  # (N,M,C,T,V)
        x = x.view(N * M, C, T, V)                 # (N*M,C,T,V)

        # ST-GCN blocks
        x = self.l1(x, A)
        x = self.l2(x, A)
        x = self.l3(x, A)
        x = self.l4(x, A)
        x = self.l5(x, A)
        x = self.l6(x, A)

        # Global average pooling over T and V
        x = x.mean(dim=-1).mean(dim=-1)            # (N*M, 256)

        # Restore (N,M,feat) and pool over M (though M=1)
        x = x.view(N, M, -1).mean(dim=1)           # (N,256)

        if return_embedding:
            return x
        return self.fc(x)                          # (N,num_classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="Path to pose_outputs.pt")
    ap.add_argument("--num_classes", type=int, default=10)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    ckpt = torch.load(args.pt, map_location="cpu")
    x_in = ckpt["x_in"]        # (1,4,60,22,1)
    A_body = ckpt["A_body"]    # (3,22,22)

    print("Loaded:")
    print(" x_in:", tuple(x_in.shape))
    print(" A_body:", tuple(A_body.shape))

    model = STGCN(in_channels=x_in.shape[1], num_classes=args.num_classes, V=x_in.shape[3], K=A_body.shape[0])
    model.to(device)
    model.eval()

    x_in = x_in.to(device)
    A_body = A_body.to(device)

    with torch.no_grad():
        logits = model(x_in, A_body)

    print("logits:", tuple(logits.shape))
    print("logits[0]:", logits[0].detach().cpu().numpy())


if __name__ == "__main__":
    main()
