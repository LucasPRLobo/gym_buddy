#!/usr/bin/env python3
"""
Batch converts a folder of exercise videos into ST-GCN-ready pose tensors.

Expected input layout:
  <videos_root>/
    squat/
      vid1.mp4
      vid2.mp4
    push-up/
      clip_a.mp4

Outputs (per video):
  <out_root>/<label>/<video_stem>/pose_outputs.pt
  <out_root>/<label>/<video_stem>/pose_outputs.npz

By default, visuals are skipped to keep the dataset compact. Enable them with
--save_visuals if you want annotated videos/plots for inspection.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple
import traceback

try:
    from tqdm import tqdm
except ImportError:  # optional dependency
    tqdm = None

from pose_stgcn_pipeline import run_pipeline


def find_videos(videos_root: Path, max_per_class: int | None) -> List[Tuple[str, Path]]:
    items: List[Tuple[str, Path]] = []
    for class_dir in sorted(videos_root.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name.lower().replace(" ", "_")
        vids = sorted(class_dir.glob("*.mp4"))
        if max_per_class is not None:
            vids = vids[:max_per_class]
        for vp in vids:
            items.append((label, vp))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_root", required=True, help="Root folder with class subdirs of .mp4 files")
    ap.add_argument("--out_root", required=True, help="Where to store extracted pose tensors")
    ap.add_argument("--t_target", type=int, default=60, help="Resampled clip length")
    ap.add_argument("--max_per_class", type=int, default=None, help="Cap videos per class (for quick runs)")
    ap.add_argument("--overwrite", action="store_true", help="Re-run even if pose_outputs.pt exists")
    ap.add_argument("--save_visuals", action="store_true", help="Keep annotated video/plots (slower, larger)")
    ap.add_argument("--no_visuals", dest="save_visuals", action="store_false",
                   help="Alias to disable visuals (default)")
    ap.set_defaults(save_visuals=False)
    args = ap.parse_args()

    videos_root = Path(args.videos_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    candidates = find_videos(videos_root, args.max_per_class)
    if not candidates:
        print(f"[ERR] No .mp4 files found under {videos_root}")
        sys.exit(1)

    print(f"[INFO] Found {len(candidates)} videos across {videos_root}")
    manifest = []

    iterable = candidates
    if tqdm is not None:
        iterable = tqdm(candidates, total=len(candidates), desc="Processing videos")

    for label, vid_path in iterable:
        rel_name = f"{label}/{vid_path.stem}"
        out_dir = out_root / rel_name
        out_dir.mkdir(parents=True, exist_ok=True)
        pt_path = out_dir / "pose_outputs.pt"

        if pt_path.exists() and not args.overwrite:
            msg = f"[SKIP] {rel_name} (pose_outputs.pt exists)"
            if tqdm is not None:
                tqdm.write(msg)
            else:
                print(msg)
            manifest.append({"label": label, "video": str(vid_path), "pt": str(pt_path)})
            continue

        msg = f"[RUN] {rel_name}"
        if tqdm is not None:
            tqdm.write(msg)
        else:
            print(msg)
        try:
            run_pipeline(
                video_path=str(vid_path),
                out_dir=str(out_dir),
                t_target=args.t_target,
                debug_ts=[],
                joint=24,
                save_visuals=args.save_visuals,
            )
            manifest.append({"label": label, "video": str(vid_path), "pt": str(pt_path)})
        except Exception as exc:
            fail_msg = f"[FAIL] {rel_name}: {exc}"
            if tqdm is not None:
                tqdm.write(fail_msg)
            else:
                print(fail_msg)
            traceback.print_exc()

    manifest_path = out_root / "dataset_index.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[DONE] Wrote manifest with {len(manifest)} entries to {manifest_path}")


if __name__ == "__main__":
    main()
