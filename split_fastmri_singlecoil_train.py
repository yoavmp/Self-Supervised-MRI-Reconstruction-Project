#!/usr/bin/env python3
import argparse
import hashlib
import os
from pathlib import Path

def stable_shuffle(paths, seed: str):
    """
    Deterministic shuffle using SHA256(seed + path).
    Works identically across machines / Python versions.
    """
    def key(p: Path):
        h = hashlib.sha256((seed + str(p)).encode("utf-8")).hexdigest()
        return h
    return sorted(paths, key=key)

def write_list(paths, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in paths:
            f.write(str(p) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", type=str, required=True,
                    help="Directory with labeled training volumes, e.g. .../singlecoil_train")
    ap.add_argument("--out-dir", type=str, required=True,
                    help="Where to write train/val/test_internal list files")
    ap.add_argument("--seed", type=str, default="yoav_seed_0",
                    help="Any string. Same string => same split.")
    ap.add_argument("--val-frac", type=float, default=0.10,
                    help="Fraction of volumes to use for validation")
    ap.add_argument("--test-frac", type=float, default=0.10,
                    help="Fraction of volumes to use for internal test")
    ap.add_argument("--pattern", type=str, default="*.h5",
                    help="Glob pattern for volumes")
    args = ap.parse_args()

    train_root = Path(args.train_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    assert train_root.exists(), f"train-root not found: {train_root}"

    files = sorted(train_root.glob(args.pattern))
    files = [p for p in files if p.is_file()]

    if len(files) == 0:
        raise RuntimeError(f"No files found in {train_root} with pattern {args.pattern}")

    # Deterministic ordering
    files = stable_shuffle(files, seed=args.seed)

    n = len(files)
    n_val = int(round(n * args.val_frac))
    n_test = int(round(n * args.test_frac))

    # Guardrails
    if n_val + n_test >= n:
        raise ValueError("val-frac + test-frac too large; leaves no train files.")

    val_files = files[:n_val]
    test_files = files[n_val:n_val + n_test]
    train_files = files[n_val + n_test:]

    # Write
    write_list(train_files, out_dir / "train_list.txt")
    write_list(val_files, out_dir / "val_list.txt")
    write_list(test_files, out_dir / "test_internal_list.txt")

    print("Done.")
    print(f"Total volumes: {n}")
    print(f"Train: {len(train_files)}")
    print(f"Val:   {len(val_files)}")
    print(f"Test_internal: {len(test_files)}")
    print(f"Lists written to: {out_dir}")

if __name__ == "__main__":
    main()
