from __future__ import annotations
import shutil
from pathlib import Path
import random
import argparse
import json
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Data root containing Images/ and StanfordExtra_V12/")
    ap.add_argument("--make-smoke", type=int, default=0, help="Create smoke subset with N images")
    args = ap.parse_args()

    root = Path(args.root)
    json_path = root / "StanfordExtra_V12/StanfordExtra_v12.json"
    assert json_path.exists(), f"Missing {json_path}"
    data = json.loads(json_path.read_text())

    ids = np.load(root / "StanfordExtra_V12/test_stanford_StanfordExtra_v12.npy")
    if args.make_smoke > 0:
        ids = ids[: args.make_smoke]

    smoke_dir = root / "stanford_smoke"
    (smoke_dir / "images").mkdir(parents=True, exist_ok=True)
    copied = 0
    for i in ids:
        rel = data[int(i)]["img_path"]
        src = root / "Images" / rel
        dst = smoke_dir / "images" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    print(f"Copied {copied} images into {smoke_dir}")

if __name__ == "__main__":
    main()
