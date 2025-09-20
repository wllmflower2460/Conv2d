
from __future__ import annotations
import json
from pathlib import Path
import click

from evaluation.oks_eval import run_eval

@click.group()
def main():
    """tcn-vae toolkit (OKS eval)."""

@main.command("eval-oks")
@click.option("--data", "data_root", type=click.Path(exists=True, file_okay=False), required=True, help="Root folder with StanfordExtra_V12 and Images")
@click.option("--split", type=click.Choice(["train", "val"]), default="val")
@click.option("--subset", type=click.Choice(["all", "smoke"]), default="smoke")
@click.option("--out", "out_dir", type=click.Path(file_okay=False), default="artifacts")
@click.option("--predictor", type=click.Choice(["yolov8", "yolo8", "yolo", "sleap"]), required=True, help="Choose predictor backend")
@click.option("--model", "model_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Model weights/path for SLEAP/YOLOv8")
@click.option("--conf", default=0.25, show_default=True, help="Box confidence threshold (YOLOv8)")
@click.option("--iou", default=0.5, show_default=True, help="NMS IoU threshold (YOLOv8)")
@click.option("--kpt-conf", default=0.5, show_default=True, help="Keypoint confidence → visibility threshold")
@click.option("--device", default=None, help="Torch device for YOLOv8 (e.g., 'cuda:0')")
def eval_oks_cmd(data_root, split, subset, out_dir, predictor, model_path, conf, iou, kpt_conf, device):
    """Evaluate OKS on Stanford Dogs + StanfordExtra with real predictors (YOLOv8/SLEAP)."""
    pred_kw = {}
    if predictor in ("yolo", "yolo8", "yolov8"):
        pred_kw.update(dict(conf=conf, iou=iou, kpt_conf=kpt_conf, device=device))
        metrics = run_eval(data_root, split, subset, out_dir, predictor_name="yolov8", model_path=model_path, **pred_kw)
    elif predictor == "sleap":
        pred_kw.update(dict(kpt_conf=kpt_conf))
        metrics = run_eval(data_root, split, subset, out_dir, predictor_name="sleap", model_path=model_path, **pred_kw)
    click.echo(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
