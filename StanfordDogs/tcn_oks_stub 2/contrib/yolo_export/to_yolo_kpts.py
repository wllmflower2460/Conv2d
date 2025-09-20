from __future__ import annotations
import json
from pathlib import Path
import numpy as np

def main(root: str | Path, out_dir: str | Path):
    root = Path(root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ann_root = root / "StanfordExtra_V12"
    data = json.loads((ann_root / "StanfordExtra_v12.json").read_text())
    ids = np.load(ann_root / "test_stanford_StanfordExtra_v12.npy")
    for i in ids[:5]:  # demo: 5 files
        j = data[int(i)]
        w, h = j["img_width"], j["img_height"]
        xmin, ymin, bw, bh = j["img_bbox"]
        # YOLO bbox: (xc, yc, w, h) normalized
        xc = (xmin + bw / 2.0) / w
        yc = (ymin + bh / 2.0) / h
        wn = bw / w
        hn = bh / h
        kpts = np.array(j["joints"], dtype=float)
        kpts[np.isnan(kpts)] = 0.0
        coords = []
        for x, y, v in kpts:
            coords += [x / w, y / h, 2.0 if v > 0 else 0.0]
        line = "0 {:.5f} {:.5f} {:.5f} {:.5f} ".format(xc, yc, wn, hn) + " ".join(f"{c:.5f}" if i%3!=2 else f"{int(c)}" for i, c in enumerate(coords))
        out_path = out_dir / (Path(j["img_path"]).stem + ".txt")
        out_path.write_text(line + "\n")

if __name__ == "__main__":
    # Example:
    # python to_yolo_kpts.py /data/root ./yolo_labels
    import sys
    main(sys.argv[1], sys.argv[2])
