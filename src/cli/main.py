#!/usr/bin/env python3
"""
Conv2d CLI - Behavioral Synchrony Analysis Pipeline
====================================================

A comprehensive command-line interface for the Conv2d-FSQ-HSMM pipeline.

Usage:
    conv2d [COMMAND] [OPTIONS]

Commands:
    preprocess  - Prepare and validate input data
    train       - Train Conv2d encoder model
    fsq-encode  - Apply FSQ quantization to features
    cluster     - Perform post-hoc clustering on codes
    smooth      - Apply temporal smoothing filters
    eval        - Evaluate model performance
    pack        - Bundle model for deployment

Exit Codes:
    0  - Success
    1  - General error
    2  - Data quality failure
    3  - Model convergence failure
    4  - Configuration error
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
import hashlib
from datetime import datetime

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich import print as rprint
from rich.panel import Panel
from rich.layout import Layout
import yaml

from .commands import (
    preprocess,
    train,
    fsq_encode,
    cluster,
    smooth,
    evaluate,
    pack
)
from ..conv2d.contracts import validate_contracts_cli, get_contract_info
from .qa_gates import QAGate, QAResult
from .utils import setup_logging, load_config

app = typer.Typer(
    name="conv2d",
    help="Conv2d-FSQ-HSMM Behavioral Synchrony Pipeline",
    no_args_is_help=True,
    rich_markup_mode="rich"
)
console = Console()

@app.callback()
def main_callback(
    version: bool = typer.Option(
        False, "--version", "-v",
        help="Show version and exit"
    ),
    verbose: bool = typer.Option(
        False, "--verbose",
        help="Enable verbose logging"
    )
):
    """Conv2d CLI for behavioral synchrony analysis."""
    if version:
        console.print(f"[cyan]Conv2d CLI v1.0.0[/cyan]")
        raise typer.Exit()
    
    if verbose:
        setup_logging(level="DEBUG")
    else:
        setup_logging(level="INFO")

@app.command()
def preprocess(
    input_dir: Path = typer.Argument(
        ...,
        help="Input directory containing raw data",
        exists=True,
        dir_okay=True
    ),
    output_dir: Path = typer.Argument(
        ...,
        help="Output directory for preprocessed data"
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Configuration file (YAML/JSON)",
        exists=True
    ),
    window_size: int = typer.Option(
        100, "--window-size", "-w",
        help="Sliding window size"
    ),
    stride: int = typer.Option(
        50, "--stride", "-s",
        help="Window stride"
    ),
    check_quality: bool = typer.Option(
        True, "--check-quality/--no-check-quality",
        help="Enable data quality checks"
    )
):
    """
    Preprocess raw IMU/behavioral data.
    
    Performs:
    - Data loading and validation
    - NaN detection and interpolation
    - Outlier detection (MAD-based)
    - Sliding window extraction
    - Normalization and scaling
    """
    console.print(Panel.fit(
        "[bold cyan]Preprocessing Pipeline[/bold cyan]\n"
        f"Input: {input_dir}\n"
        f"Output: {output_dir}\n"
        f"Window: {window_size} @ stride {stride}",
        title="🔧 Preprocess"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("[cyan]Loading data...", total=100)
        
        # Load configuration
        cfg = load_config(config) if config else {}
        cfg.update({
            "window_size": window_size,
            "stride": stride,
            "check_quality": check_quality
        })
        
        progress.update(task, advance=20, description="[cyan]Validating data...")
        
        # QA Gate: Data quality check
        qa_gate = QAGate("data_quality")
        qa_result = preprocess.validate_data(input_dir, cfg)
        
        if not qa_result.passed and check_quality:
            console.print(f"[red]❌ Data quality check failed:[/red]")
            for issue in qa_result.issues:
                console.print(f"  • {issue}")
            raise typer.Exit(code=2)
        
        progress.update(task, advance=30, description="[cyan]Extracting windows...")
        
        # Process data
        stats = preprocess.process_data(
            input_dir, output_dir, cfg, progress
        )
        
        progress.update(task, advance=50, description="[cyan]Complete!")
    
    # Display results table
    table = Table(title="Preprocessing Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Windows", str(stats["total_windows"]))
    table.add_row("Valid Windows", str(stats["valid_windows"]))
    table.add_row("NaN Events", str(stats["nan_events"]))
    table.add_row("Outliers Detected", str(stats["outliers"]))
    table.add_row("Processing Time", f"{stats['time']:.2f}s")
    
    console.print(table)
    console.print(f"[green]✓ Preprocessing complete![/green]")

@app.command()
def train(
    data_dir: Path = typer.Argument(
        ...,
        help="Directory with preprocessed data",
        exists=True
    ),
    output_dir: Path = typer.Argument(
        ...,
        help="Output directory for models"
    ),
    epochs: int = typer.Option(
        100, "--epochs", "-e",
        help="Number of training epochs"
    ),
    batch_size: int = typer.Option(
        32, "--batch-size", "-b",
        help="Training batch size"
    ),
    lr: float = typer.Option(
        1e-3, "--learning-rate", "-lr",
        help="Initial learning rate"
    ),
    architecture: str = typer.Option(
        "conv2d-vq", "--arch", "-a",
        help="Model architecture",
        callback=lambda x: x if x in ["conv2d-vq", "conv2d-fsq", "tcn-vae"] else typer.BadParameter(f"Unknown architecture: {x}")
    )
):
    """
    Train Conv2d encoder model.
    
    Supports:
    - Conv2d-VQ with EMA codebook
    - Conv2d-FSQ with finite quantization
    - TCN-VAE baseline
    """
    console.print(Panel.fit(
        f"[bold cyan]Training {architecture.upper()}[/bold cyan]\n"
        f"Data: {data_dir}\n"
        f"Epochs: {epochs} | Batch: {batch_size} | LR: {lr}",
        title="🚀 Training"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        
        main_task = progress.add_task(
            f"[cyan]Training {architecture}...", 
            total=epochs
        )
        
        # Training loop with rich progress
        best_acc = 0.0
        for epoch in range(epochs):
            metrics = train.train_epoch(
                data_dir, architecture, epoch, 
                batch_size, lr
            )
            
            # Update progress
            progress.update(
                main_task, 
                advance=1,
                description=f"[cyan]Epoch {epoch+1}/{epochs} | Loss: {metrics['loss']:.4f} | Acc: {metrics['acc']:.2%}"
            )
            
            if metrics['acc'] > best_acc:
                best_acc = metrics['acc']
                train.save_checkpoint(output_dir, architecture, metrics)
        
        # QA Gate: Convergence check
        if best_acc < 0.60:
            console.print(f"[red]❌ Model failed to converge (acc={best_acc:.2%} < 60%)[/red]")
            raise typer.Exit(code=3)
    
    console.print(f"[green]✓ Training complete! Best accuracy: {best_acc:.2%}[/green]")

@app.command(name="fsq-encode")
def fsq_encode_cmd(
    model_path: Path = typer.Argument(
        ...,
        help="Path to trained model",
        exists=True
    ),
    data_dir: Path = typer.Argument(
        ...,
        help="Input data directory",
        exists=True
    ),
    output_file: Path = typer.Argument(
        ...,
        help="Output file for FSQ codes"
    ),
    levels: str = typer.Option(
        "4,4,4", "--levels", "-l",
        help="FSQ quantization levels (comma-separated)"
    )
):
    """
    Apply FSQ quantization to extract discrete codes.
    
    Generates:
    - Discrete behavioral codes
    - Code usage statistics
    - Perplexity metrics
    """
    levels_list = [int(x) for x in levels.split(",")]
    total_codes = 1
    for l in levels_list:
        total_codes *= l
    
    console.print(Panel.fit(
        f"[bold cyan]FSQ Encoding[/bold cyan]\n"
        f"Levels: {levels_list} → {total_codes} codes\n"
        f"Model: {model_path.name}",
        title="🔢 FSQ Quantization"
    ))
    
    with console.status("[cyan]Encoding features...") as status:
        results = fsq_encode.encode_dataset(
            model_path, data_dir, levels_list
        )
        
        # Save codes
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'wb') as f:
            import pickle
            pickle.dump(results, f)
    
    # Display code usage table
    table = Table(title="Code Usage Statistics")
    table.add_column("Code ID", style="cyan")
    table.add_column("Count", style="yellow")
    table.add_column("Frequency", style="green")
    
    for code_id, count in results['code_counts'].most_common(10):
        freq = count / results['total_samples']
        table.add_row(str(code_id), str(count), f"{freq:.1%}")
    
    console.print(table)
    console.print(f"[green]Perplexity: {results['perplexity']:.2f}[/green]")
    console.print(f"[green]Active Codes: {len(results['code_counts'])}/{total_codes}[/green]")

@app.command()
def cluster(
    codes_file: Path = typer.Argument(
        ...,
        help="FSQ codes file from encode step",
        exists=True
    ),
    output_dir: Path = typer.Argument(
        ...,
        help="Output directory for clusters"
    ),
    n_clusters: int = typer.Option(
        12, "--n-clusters", "-k",
        help="Number of behavioral clusters"
    ),
    method: str = typer.Option(
        "kmeans", "--method", "-m",
        help="Clustering method",
        callback=lambda x: x if x in ["kmeans", "gmm", "spectral"] else typer.BadParameter(f"Unknown method: {x}")
    ),
    min_support: float = typer.Option(
        0.005, "--min-support",
        help="Minimum cluster support (fraction)"
    )
):
    """
    Post-hoc clustering of FSQ codes into behavioral motifs.
    
    Methods:
    - K-means (fast, deterministic)
    - GMM (probabilistic)
    - Spectral (graph-based)
    """
    console.print(Panel.fit(
        f"[bold cyan]Behavioral Clustering[/bold cyan]\n"
        f"Method: {method.upper()} with k={n_clusters}\n"
        f"Min support: {min_support:.1%}",
        title="🎯 Clustering"
    ))
    
    with console.status("[cyan]Clustering codes...") as status:
        results = cluster.cluster_codes(
            codes_file, method, n_clusters, min_support
        )
        
        # Save clusters
        output_dir.mkdir(parents=True, exist_ok=True)
        cluster.save_clusters(output_dir, results)
    
    # Display motif statistics
    table = Table(title="Behavioral Motifs")
    table.add_column("Motif", style="cyan")
    table.add_column("Samples", style="yellow")
    table.add_column("Frequency", style="green")
    table.add_column("Mean Duration", style="magenta")
    
    for motif_id in range(n_clusters):
        mask = results['labels'] == motif_id
        count = mask.sum()
        if count > 0:
            freq = count / len(results['labels'])
            duration = results['durations'].get(motif_id, 0)
            table.add_row(
                f"Motif {motif_id}", 
                str(count),
                f"{freq:.1%}",
                f"{duration:.1f} frames"
            )
    
    console.print(table)
    console.print(f"[green]✓ Clustering complete![/green]")

@app.command()
def smooth(
    clusters_dir: Path = typer.Argument(
        ...,
        help="Directory with cluster assignments",
        exists=True
    ),
    output_dir: Path = typer.Argument(
        ...,
        help="Output directory for smoothed labels"
    ),
    window: int = typer.Option(
        7, "--window", "-w",
        help="Median filter window size"
    ),
    min_duration: int = typer.Option(
        3, "--min-duration", "-d",
        help="Minimum motif duration (frames)"
    )
):
    """
    Apply temporal smoothing to behavioral sequences.
    
    Filters:
    - Median filtering for noise reduction
    - Minimum duration enforcement
    - Transition probability smoothing
    """
    console.print(Panel.fit(
        f"[bold cyan]Temporal Smoothing[/bold cyan]\n"
        f"Window: {window} frames\n"
        f"Min duration: {min_duration} frames",
        title="🌊 Smoothing"
    ))
    
    with console.status("[cyan]Applying temporal filters...") as status:
        results = smooth.apply_smoothing(
            clusters_dir, window, min_duration
        )
        
        # Save smoothed labels
        output_dir.mkdir(parents=True, exist_ok=True)
        smooth.save_smoothed(output_dir, results)
    
    # Show before/after statistics
    table = Table(title="Smoothing Effects")
    table.add_column("Metric", style="cyan")
    table.add_column("Before", style="yellow")
    table.add_column("After", style="green")
    
    table.add_row("Transitions", str(results['transitions_before']), str(results['transitions_after']))
    table.add_row("Mean Duration", f"{results['duration_before']:.1f}", f"{results['duration_after']:.1f}")
    table.add_row("Noise Events", str(results['noise_before']), str(results['noise_after']))
    
    console.print(table)
    console.print(f"[green]✓ Smoothing reduced transitions by {results['reduction']:.1%}[/green]")

@app.command(name="eval")
def evaluate_cmd(
    model_dir: Path = typer.Argument(
        ...,
        help="Directory with trained model",
        exists=True
    ),
    test_data: Path = typer.Argument(
        ...,
        help="Test data directory",
        exists=True
    ),
    output_dir: Path = typer.Argument(
        ...,
        help="Output directory for evaluation"
    ),
    metrics: str = typer.Option(
        "all", "--metrics", "-m",
        help="Metrics to compute (all/basic/extended)"
    )
):
    """
    Evaluate model performance with comprehensive metrics.
    
    Metrics:
    - Accuracy, Precision, Recall, F1
    - Expected Calibration Error (ECE)
    - Mutual Information I(Z;Φ)
    - Behavioral transition matrices
    """
    console.print(Panel.fit(
        "[bold cyan]Model Evaluation[/bold cyan]\n"
        f"Model: {model_dir.name}\n"
        f"Test data: {test_data.name}",
        title="📊 Evaluation"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("[cyan]Evaluating model...", total=4)
        
        # Run evaluation
        results = evaluate.run_evaluation(
            model_dir, test_data, metrics, progress, task
        )
        
        # Generate evaluation bundle
        exp_hash = hashlib.md5(
            f"{model_dir}_{test_data}_{datetime.now()}".encode()
        ).hexdigest()[:8]
        
        output_dir.mkdir(parents=True, exist_ok=True)
        evaluate.save_bundle(output_dir / f"eval_{exp_hash}", results)
    
    # Display main metrics table
    table = Table(title=f"Evaluation Results [EXP_{exp_hash}]")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Target", style="yellow")
    table.add_column("Status", style="bold")
    
    metrics_display = [
        ("Accuracy", results['accuracy'], 0.75, results['accuracy'] >= 0.75),
        ("Macro F1", results['macro_f1'], 0.70, results['macro_f1'] >= 0.70),
        ("ECE", results['ece'], 0.10, results['ece'] <= 0.10),
        ("Perplexity", results['perplexity'], 5.0, results['perplexity'] <= 5.0),
        ("Coverage", results['coverage'], 1.0, results['coverage'] >= 0.95),
    ]
    
    for name, value, target, passed in metrics_display:
        status = "✓" if passed else "✗"
        color = "green" if passed else "red"
        table.add_row(
            name,
            f"{value:.3f}",
            f"{target:.3f}",
            f"[{color}]{status}[/{color}]"
        )
    
    console.print(table)
    
    # Display behavioral metrics
    if 'behavioral_metrics' in results:
        behav_table = Table(title="Behavioral Metrics")
        behav_table.add_column("Metric", style="cyan")
        behav_table.add_column("Value", style="magenta")
        
        behav_table.add_row("Mean Dwell Time", f"{results['behavioral_metrics']['mean_dwell']:.1f} frames")
        behav_table.add_row("Transition Rate", f"{results['behavioral_metrics']['transition_rate']:.3f}")
        behav_table.add_row("Motif Diversity", f"{results['behavioral_metrics']['diversity']:.2f}")
        
        console.print(behav_table)
    
    # QA Gate check
    all_passed = all(p for _, _, _, p in metrics_display)
    if not all_passed:
        console.print("[yellow]⚠ Some metrics below target thresholds[/yellow]")
    else:
        console.print(f"[green]✓ All quality gates passed![/green]")
    
    console.print(f"[cyan]Evaluation bundle saved: eval_{exp_hash}/[/cyan]")

@app.command()
def pack(
    config: Path = typer.Argument(
        ...,
        help="Configuration file (YAML)",
        exists=True
    ),
    metrics: Path = typer.Argument(
        ...,
        help="Metrics file (JSON)",
        exists=True
    ),
    model: Path = typer.Argument(
        ...,
        help="Model file or directory",
        exists=True
    ),
    format: str = typer.Option(
        "onnx", "--format", "-f",
        help="Model format (onnx/coreml/hailo)"
    ),
    output_dir: Path = typer.Option(
        Path("artifacts"), "--output-dir", "-o",
        help="Output directory for artifact bundle"
    ),
    name: Optional[str] = typer.Option(
        None, "--name", "-n",
        help="Bundle name (optional)"
    )
):
    """
    Create standardized artifact bundle for deployment.
    
    Creates artifacts/EXP_HASH/ directory with:
    - config.yaml (experiment configuration)
    - metrics.json (evaluation results)
    - label_map.json (frozen behavioral labels)
    - model.onnx/coreml.mlpackage/hailo.hef (model file)
    - VERSION (git version)
    - COMMIT_SHA (git commit)
    """
    console.print(Panel.fit(
        f"[bold cyan]Artifact Bundle[/bold cyan]\n"
        f"Config: {config.name}\n"
        f"Metrics: {metrics.name}\n"
        f"Model: {model.name} ({format})\n"
        f"Output: {output_dir}/",
        title="📦 Packaging"
    ))
    
    with console.status("[cyan]Creating artifact bundle...") as status:
        try:
            # Create bundle
            bundle = pack.create_bundle(
                config_path=config,
                metrics_path=metrics,
                model_path=model,
                model_format=format,
                output_dir=output_dir,
                bundle_name=name
            )
            
            # Get bundle info
            bundle_info = bundle.get_info()
            
        except Exception as e:
            console.print(f"[red]❌ Bundle creation failed: {e}[/red]")
            raise typer.Exit(code=1)
    
    # Display bundle contents
    table = Table(title="Bundle Contents")
    table.add_column("File", style="cyan")
    table.add_column("Size", style="yellow")
    table.add_column("Type", style="dim")
    
    for filename, info in bundle_info['files'].items():
        if isinstance(info, dict) and 'size' in info:
            size = pack.format_size(info['size'])
            file_type = info.get('type', 'file')
        else:
            size = "-"
            file_type = info.get('type', 'directory')
        
        table.add_row(filename, size, file_type)
    
    console.print(table)
    
    # Display deployment info
    model_formats_str = ", ".join(bundle_info['model_formats'])
    info_panel = Panel.fit(
        f"[bold]Artifact Bundle Ready![/bold]\n\n"
        f"Bundle Hash: {bundle.bundle_hash}\n"
        f"Directory: {bundle.bundle_dir}\n"
        f"Model Formats: {model_formats_str}\n"
        f"Created: {bundle_info.get('created_at', 'unknown')}\n\n"
        f"[dim]Verify with:[/dim]\n"
        f"  conv2d pack verify {bundle.bundle_hash}\n\n"
        f"[dim]Archive with:[/dim]\n"
        f"  conv2d pack archive {bundle.bundle_hash}",
        title="🚀 Bundle Info",
        border_style="green"
    )
    console.print(info_panel)
    
    console.print(f"[green]✓ Bundle created: {bundle.bundle_hash}[/green]")

@app.command(name="validate-contracts")
def validate_contracts(
    input_file: Path = typer.Argument(
        ...,
        help="Input file to validate (.pt or .npy)",
        exists=True
    ),
    strict: bool = typer.Option(
        True, "--strict/--no-strict",
        help="Enable strict validation mode"
    )
):
    """
    Validate pipeline contracts against test data.
    
    Validates that input data conforms to the FROZEN pipeline contracts
    defined in docs/contracts.md. This ensures compatibility with
    production deployment and prevents interface drift.
    """
    console.print(Panel.fit(
        "[bold cyan]Contract Validation[/bold cyan]\n"
        f"Input: {input_file}\n"
        f"Strict: {'Yes' if strict else 'No'}",
        title="🔒 Contract Enforcement"
    ))
    
    success = validate_contracts_cli(str(input_file), strict=strict)
    
    if success:
        console.print("[green]✅ All contracts validated successfully![/green]")
    else:
        console.print("[red]❌ Contract validation failed![/red]")
        raise typer.Exit(code=4)  # Configuration/contract error

@app.command(name="contract-info")
def contract_info():
    """
    Display information about frozen pipeline contracts.
    
    Shows the current contract specifications including shapes,
    data types, and validation rules. All contracts are FROZEN
    for production stability.
    """
    console.print(Panel.fit(
        "[bold cyan]Pipeline Contracts[/bold cyan]\n"
        "Displaying FROZEN contract specifications",
        title="🔒 Contract Information"
    ))
    
    info = get_contract_info()
    
    # Display contract summary
    table = Table(title="Contract Summary")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green") 
    table.add_column("Version", style="yellow")
    
    table.add_row("Input", "FROZEN", info["version"])
    table.add_row("FSQ Encoding", "FROZEN", info["version"])
    table.add_row("Clustering", "FROZEN", info["version"])
    table.add_row("Temporal Smoothing", "FROZEN", info["version"])
    table.add_row("Label Mapping", "FROZEN", info["version"])
    
    console.print(table)
    
    # Display key specifications
    specs_panel = Panel.fit(
        f"[bold]Key Specifications:[/bold]\n\n"
        f"Input Shape: {info['contracts']['input']['shape']}\n"
        f"Input Dtype: {info['contracts']['input']['dtype']}\n"
        f"FSQ Levels: {info['contracts']['fsq']['levels']}\n"
        f"Codebook Size: {info['contracts']['fsq']['codebook_size']}\n"
        f"Default Clusters: {info['contracts']['clustering']['default_clusters']}\n"
        f"Label Cardinality: {info['contracts']['labels']['cardinality']}\n"
        f"Validation Enabled: {'Yes' if info['validation_enabled'] else 'No'}\n\n"
        f"[dim]See docs/contracts.md for complete specification[/dim]",
        title="📋 Contract Details",
        border_style="blue"
    )
    console.print(specs_panel)

if __name__ == "__main__":
    app()