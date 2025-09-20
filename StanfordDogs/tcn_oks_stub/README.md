# tcn-vae-oks (stub)

Scaffold to evaluate **Object Keypoint Similarity (OKS)** using **Stanford Dogs + StanfordExtra**.
Pairs with your `tcn-vae-training-pipeline`. Drop this into your repo or merge files as needed.

## Quick start

```bash
pip install -e .[dev]
python cli/eval_oks.py --data ./data/stanford_extra --split val --subset smoke --out artifacts
```
