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
@click.option("--predictor", type=click.Choice(["passthrough", "gaussian"]), default="passthrough")
def eval_oks_cmd(data_root, split, subset, out_dir, predictor):
    """Evaluate OKS on Stanford Dogs + StanfordExtra with a simple predictor stub."""
    metrics = run_eval(data_root, split, subset, out_dir, predictor_kind=predictor)
    click.echo(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
