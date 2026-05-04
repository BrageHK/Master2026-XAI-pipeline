#!/usr/bin/env python3
import sys
import os
import numpy as np

def inspect_folder(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith('.npz')])
    if not files:
        print(f"No .npz files in {folder}")
        return

    print(f"\n=== {folder} ({len(files)} files) ===")
    all_keys = {}

    for fname in files:
        path = os.path.join(folder, fname)
        data = np.load(path, allow_pickle=True)
        for k in data.keys():
            val = data[k]
            if k not in all_keys:
                all_keys[k] = set()
            if val.ndim == 0:
                all_keys[k].add(str(val))
            else:
                all_keys[k].add(f"shape={val.shape} dtype={val.dtype}")

    for k, variants in sorted(all_keys.items()):
        if len(variants) == 1:
            print(f"  {k}: {next(iter(variants))}")
        else:
            print(f"  {k}: [{len(variants)} variants]")
            for v in sorted(variants):
                print(f"    - {v}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_npz.py <folder> [folder2 ...]")
        sys.exit(1)
    for folder in sys.argv[1:]:
        inspect_folder(folder)