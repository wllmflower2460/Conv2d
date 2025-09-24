"""
Ablation Framework with Complete VQ Fixes
Based on VQ_Codebook_Collapse_Analysis.md recommendations
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import time
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def mixup_batch(batch_x, batch_y, alpha=1.0):
    if alpha <= 0:
        return batch_x, batch_y
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(batch_x.size(0), device=batch_x.device)
    mixed_x = lam * batch_x + (1 - lam) * batch_x[idx]
    y_a, y_b = batch_y, batch_y[idx]
    return mixed_x, (y_a, y_b, lam)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, lr=1e-2, max_iter=100):
    temperature = nn.Parameter(torch.ones(1, device=logits.device))
    optimizer = optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
    def eval_loss():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss
    optimizer.step(eval_loss)
    return temperature.item()


def expected_calibration_error(confidences: np.ndarray, accuracies: np.ndarray, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece


def conformal_threshold(probs: np.ndarray, labels: np.ndarray, alpha: float = 0.1):
    n = len(labels)
    nc = []
    for i in range(n):
        nc.append(1.0 - probs[i, labels[i]])
    nc = np.array(nc)
    try:
        q = np.quantile(nc, 1 - alpha, method="higher")
    except TypeError:
        q = np.quantile(nc, 1 - alpha)
    return float(q)


def prediction_set_mask(probs: np.ndarray, q: float) -> np.ndarray:
    """
    Return boolean mask (N, C) of labels included in each prediction set.
    Guarantee non-empty set by forcing inclusion of the argmax label.
    """
    mask = (1.0 - probs) <= q
    if mask.ndim != 2:
        raise ValueError("probs must be 2D (N, C)")
    top1 = probs.argmax(axis=1)
    mask[np.arange(mask.shape[0]), top1] = True
    return mask


@dataclass
class AblationConfig:
    use_vq: bool = False
    use_hdp: bool = False
    use_hsmm: bool = False
    use_entropy: bool = False
    use_aug: bool = False
    num_vq_codes: int = 512
    commitment_cost: float = 0.25
    hdp_clusters: int = 50
    hsmm_states: int = 25


@dataclass
class AblationResult:
    config_name: str
    accuracy: float
    loss: float
    perplexity: float
    ece: float
    inference_time_ms_p50: float
    inference_time_ms_p95: float
    num_parameters: int
    memory_mb: float
    convergence_epoch: int = 0
    pi90_coverage: float = 0.0
    avg_pred_set_size: float = 0.0


class AblationModel(nn.Module):
    def __init__(self, config: AblationConfig):
        super().__init__()
        self.config = config
        from models.conv2d_vq_model import Conv2dEncoder
        self.encoder = Conv2dEncoder(
            input_channels=9, input_height=2,
            hidden_channels=[64, 128, 256],
            code_dim=64, dropout=0.2
        )
        self.vq = None
        if config.use_vq:
            # Use the stable VQ implementation with dead code reinit
            from models.VectorQuantizerEMA2D_Stable import VectorQuantizerEMA2D_Stable
            self.vq = VectorQuantizerEMA2D_Stable(
                num_codes=config.num_vq_codes, code_dim=64,
                commitment_cost=config.commitment_cost,
                decay=0.95,  # Slightly lower decay for faster adaptation
                l2_normalize_input=True,  # Normalize features before VQ
                restart_dead_codes=True,  # Critical for preventing collapse
                dead_code_threshold=1e-3
            )
        self.hdp = None
        if config.use_hdp:
            from models.hdp_components import HDPLayer
            self.hdp = HDPLayer(input_dim=64, max_clusters=config.hdp_clusters, concentration=1.0)
        self.hsmm = None
        if config.use_hsmm:
            from models.hsmm_components import HSMM
            self.hsmm = HSMM(
                num_states=config.hsmm_states,
                observation_dim=64,
                max_duration=50,
                duration_dist='negative_binomial'
            )
        self.entropy = None
        if config.use_entropy:
            from models.entropy_uncertainty import EntropyUncertaintyModule
            self.entropy = EntropyUncertaintyModule(
                state_dim=64,
                num_states=config.hsmm_states if config.use_hsmm else 12,
                num_phase_bins=12
            )
        self.classifier = nn.Linear(64, 12)

    def forward(self, x: torch.Tensor) -> Dict:
        z_e = self.encoder(x)
        if self.vq is not None:
            z_q, vq_loss, vq_info = self.vq(z_e)
            features = z_q
        else:
            features = z_e
            vq_loss = {"vq": torch.tensor(0.0, device=x.device)}
            vq_info = {"perplexity": 0.0, "usage": 0.0}
        features = features.squeeze(2)
        if self.hdp is not None:
            cluster_probs = self.hdp(features.permute(0, 2, 1))
        else:
            cluster_probs = None
        if self.hsmm is not None:
            state_seq, _ = self.hsmm(features.permute(0, 2, 1))
        else:
            state_seq = None
        logits = self.classifier(features.mean(dim=2))
        uncertainty = {}
        if self.entropy is not None and state_seq is not None:
            uncertainty = self.entropy(state_seq, features.permute(0, 2, 1))
        return {"logits": logits, "vq_loss": vq_loss, "vq_info": vq_info, "uncertainty": uncertainty}


def initialize_codebook_from_data(model: AblationModel, data_loader, device):
    """Initialize VQ codebook using actual encoder outputs"""
    if not hasattr(model, 'vq') or model.vq is None:
        return
    
    print("Initializing VQ codebook from data...")
    model.eval()
    with torch.no_grad():
        embeddings = []
        for x, _ in data_loader:
            x = x.to(device)
            z_e = model.encoder(x)
            # Reshape to (B*T, D)
            if z_e.dim() == 4:
                B, D, _, T = z_e.shape
                z_e_flat = z_e.squeeze(2).permute(0, 2, 1).reshape(-1, D)
            else:
                B, D, T = z_e.shape
                z_e_flat = z_e.permute(0, 2, 1).reshape(-1, D)
            
            # Normalize if needed
            if model.vq.l2_normalize_input:
                z_e_flat = F.normalize(z_e_flat, dim=1)
            
            embeddings.append(z_e_flat)
            if len(embeddings) * z_e_flat.size(0) >= model.vq.num_codes * 10:
                break
        
        embeddings = torch.cat(embeddings, dim=0)
        # K-means initialization or random sampling
        indices = torch.randperm(embeddings.size(0))[:model.vq.num_codes]
        model.vq.embedding.data = embeddings[indices].to(device)
        model.vq.ema_cluster_size.data.fill_(1.0)
        print(f"Initialized {model.vq.num_codes} codes from {embeddings.size(0)} samples")


class AblationRunner:
    def __init__(self, train_loader, val_loader, device="cuda", output_dir="ablation_results"):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def train_model(self, model: AblationModel, config: AblationConfig, epochs=100, seed=42):
        set_all_seeds(seed)
        model.to(self.device)
        
        # Initialize codebook from data
        initialize_codebook_from_data(model, self.train_loader, self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        max_steps = len(self.train_loader) * epochs
        steps = 0
        best_acc, conv_epoch = 0.0, 0
        
        for epoch in range(epochs):
            model.train()
            
            # VQ loss warmup
            vq_weight = min(1.0, epoch / 10) if config.use_vq else 0.0
            
            for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                if config.use_aug and epoch > 5:
                    batch_x, batch_y = mixup_batch(batch_x, batch_y, alpha=1.0)
                
                optimizer.zero_grad()
                out = model(batch_x)
                
                if isinstance(batch_y, tuple):
                    y_a, y_b, lam = batch_y
                    loss = lam * criterion(out["logits"], y_a) + (1 - lam) * criterion(out["logits"], y_b)
                else:
                    loss = criterion(out["logits"], batch_y)
                
                # Critical: Properly weighted VQ loss
                if "vq_loss" in out and isinstance(out["vq_loss"], dict) and "vq" in out["vq_loss"]:
                    loss = loss + vq_weight * 0.1 * out["vq_loss"]["vq"]
                
                loss.backward()
                optimizer.step()
                steps += 1
                
                if steps >= max_steps:
                    break
            
            # Diagnostic monitoring for VQ
            if config.use_vq and epoch % 10 == 0:
                with torch.no_grad():
                    usage = model.vq.ema_cluster_size / model.vq.ema_cluster_size.sum()
                    active_codes = (usage > 0.001).sum().item()
                    max_usage = usage.max().item()
                    perplexity = torch.exp(-torch.sum(usage * torch.log(usage + 1e-10))).item()
                    print(f"Epoch {epoch}: Active codes: {active_codes}/{model.vq.num_codes}, "
                          f"Max usage: {max_usage:.3f}, Perplexity: {perplexity:.1f}")
            
            acc = self.quick_val_accuracy(model)
            if acc > best_acc:
                best_acc, conv_epoch = acc, epoch
            
            if steps >= max_steps:
                break
        
        return best_acc, conv_epoch

    def quick_val_accuracy(self, model: AblationModel) -> float:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for bx, by in self.val_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                out = model(bx)
                pred = out["logits"].argmax(dim=1)
                correct += (pred == by).sum().item()
                total += by.size(0)
        return correct / max(total, 1)

    def evaluate_model(self, model: AblationModel, config: AblationConfig) -> AblationResult:
        model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0
        all_times_ms = []
        all_logits, all_labels = [], []
        perplexity_sum, perplexity_batches = 0, 0
        
        for bx, by in self.val_loader:
            bx, by = bx.to(self.device), by.to(self.device)
            torch.cuda.synchronize() if self.device.type == "cuda" else None
            start_time = time.perf_counter()
            with torch.no_grad():
                out = model(bx)
            torch.cuda.synchronize() if self.device.type == "cuda" else None
            elapsed_ms = 1000 * (time.perf_counter() - start_time)
            all_times_ms.append(elapsed_ms / bx.size(0))
            logits = out["logits"]
            loss = criterion(logits, by)
            total_loss += loss.item()
            all_logits.append(logits.detach().cpu())
            all_labels.append(by.detach().cpu())
            if config.use_vq and "vq_info" in out and "perplexity" in out["vq_info"]:
                perplexity_sum += float(out["vq_info"]["perplexity"])
                perplexity_batches += 1
        
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        preds = logits.argmax(dim=1)
        accuracy = float((preds == labels).float().mean().item())
        avg_loss = float(total_loss / max(len(self.val_loader), 1))
        
        # Calibration
        CAL_SPLIT = 0.5
        n = labels.size(0)
        n_cal = max(int(CAL_SPLIT * n), 1)
        logits_cal, labels_cal = logits[:n_cal], labels[:n_cal]
        logits_test, labels_test = logits[n_cal:], labels[n_cal:] if n - n_cal > 0 else (logits[:], labels[:])
        T = fit_temperature(logits_cal.to(self.device), labels_cal.to(self.device))
        probs_test = torch.softmax(logits_test / T, dim=1).cpu().numpy()
        conf_test = probs_test.max(axis=1)
        pred_test = probs_test.argmax(axis=1)
        correct_test = (pred_test == labels_test.cpu().numpy()).astype(float)
        ece = expected_calibration_error(conf_test, correct_test, n_bins=15)
        
        # Conformal prediction
        probs_cal = torch.softmax(logits_cal / T, dim=1).cpu().numpy()
        y_cal = labels_cal.cpu().numpy()
        q = conformal_threshold(probs_cal, y_cal, alpha=0.10)
        P = prediction_set_mask(probs_test, q)
        y_test = labels_test.cpu().numpy()
        pi90_coverage = float(P[np.arange(P.shape[0]), y_test].mean()) if P.shape[0] > 0 else 0.0
        avg_set_size = float(P.sum(axis=1).mean()) if P.shape[0] > 0 else 0.0
        
        # Timing
        p50_ms = float(np.percentile(all_times_ms, 50))
        p95_ms = float(np.percentile(all_times_ms, 95))
        
        # Perplexity
        perplexity = perplexity_sum / max(perplexity_batches, 1) if config.use_vq else 0.0
        
        # Memory
        if self.device.type == "cuda":
            mem_mb = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
        else:
            mem_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return AblationResult(
            config_name="",
            accuracy=accuracy,
            loss=avg_loss,
            perplexity=perplexity,
            ece=ece,
            inference_time_ms_p50=p50_ms,
            inference_time_ms_p95=p95_ms,
            num_parameters=num_params,
            memory_mb=mem_mb,
            convergence_epoch=0,
            pi90_coverage=pi90_coverage,
            avg_pred_set_size=avg_set_size,
        )

    def run_ablation(self, epochs=100, seed=123):
        configs = [
            ("baseline_encoder", AblationConfig()),
            ("vq_only", AblationConfig(use_vq=True)),
            ("hdp_only", AblationConfig(use_hdp=True)),
            ("hsmm_only", AblationConfig(use_hsmm=True)),
            ("vq_hdp", AblationConfig(use_vq=True, use_hdp=True)),
            ("vq_hsmm", AblationConfig(use_vq=True, use_hsmm=True)),
            ("hdp_hsmm", AblationConfig(use_hdp=True, use_hsmm=True)),
            ("vq_hdp_hsmm", AblationConfig(use_vq=True, use_hdp=True, use_hsmm=True)),
            ("full_no_aug", AblationConfig(use_vq=True, use_hdp=True, use_hsmm=True, use_entropy=True)),
            ("full_with_aug", AblationConfig(use_vq=True, use_hdp=True, use_hsmm=True, use_entropy=True, use_aug=True)),
            ("vq_256_codes_beta_0_4", AblationConfig(use_vq=True, use_hdp=True, use_hsmm=True, use_entropy=True, use_aug=True, num_vq_codes=256, commitment_cost=0.4)),
            ("vq_128_codes_beta_0_5", AblationConfig(use_vq=True, use_hdp=True, use_hsmm=True, use_entropy=True, use_aug=True, num_vq_codes=128, commitment_cost=0.5)),
            ("vq_512_codes_beta_0_25", AblationConfig(use_vq=True, use_hdp=True, use_hsmm=True, use_entropy=True, use_aug=True, num_vq_codes=512, commitment_cost=0.25)),
        ]
        
        results = []
        print("=" * 80)
        print("Starting Ablation Study with Complete VQ Fixes")
        print("=" * 80)
        
        for idx, (name, config) in enumerate(configs, 1):
            print(f"\n[{idx}/{len(configs)}] {name} — {self.describe_config(config)}")
            print(f"Components: VQ={config.use_vq}, HDP={config.use_hdp}, HSMM={config.use_hsmm}, "
                  f"Entropy={config.use_entropy}, Aug={config.use_aug}")
            
            model = AblationModel(config)
            print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
            
            print("Training (fixed budget)...")
            best_acc, conv_epoch = self.train_model(model, config, epochs, seed + idx)
            print(f"Best quick-val accuracy: {best_acc:.4f} @ epoch {conv_epoch}")
            
            print("Evaluating...")
            result = self.evaluate_model(model, config)
            result.config_name = name
            result.convergence_epoch = conv_epoch
            results.append(result)
            
            self.save_results(results)
            print(f"Acc={result.accuracy:.4f}  ECE={result.ece:.4f}  "
                  f"Cov90={result.pi90_coverage:.3f}  Set={result.avg_pred_set_size:.2f}  "
                  f"p95={result.inference_time_ms_p95:.2f}ms  Perplexity={result.perplexity:.1f}")
        
        return results

    def describe_config(self, config: AblationConfig) -> str:
        parts = []
        if not any([config.use_vq, config.use_hdp, config.use_hsmm, config.use_entropy]):
            return "Encoder + Classifier only"
        if config.use_vq:
            parts.append("VQ")
        if config.use_hdp:
            parts.append("HDP")
        if config.use_hsmm:
            parts.append("HSMM")
        if config.use_entropy:
            parts.append("Entropy")
        if config.use_aug:
            parts.append("Aug")
        
        if config.num_vq_codes != 512 or config.commitment_cost != 0.25:
            parts.append(f"(codes={config.num_vq_codes}, β={config.commitment_cost})")
        
        return " + ".join(parts) if parts else "Baseline"

    def save_results(self, results: List[AblationResult]):
        json_path = self.output_dir / "ablation_results.json"
        csv_path = self.output_dir / "ablation_results.csv"
        
        # JSON
        data = {r.config_name: asdict(r) for r in results}
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        
        # CSV
        import csv
        with open(csv_path, "w", newline="") as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
                writer.writeheader()
                for r in results:
                    writer.writerow(asdict(r))
        
        print(f"Saved: {json_path} and {csv_path}")

    def generate_report(self, results: List[AblationResult]):
        report = self.output_dir / "ablation_report.md"
        with open(report, "w") as f:
            f.write("# Ablation Study Report with VQ Fixes\n\n")
            f.write("**Decision line:** Pass if Accuracy ≥ 0.85, ECE ≤ 0.03, 90% coverage ∈ [0.88, 0.92], p95 ≤ 100 ms.\n\n")
            
            best = max(results, key=lambda r: r.accuracy)
            f.write(f"**Best by accuracy**: `{best.config_name}` — ")
            f.write(f"Acc {best.accuracy:.3f}, ECE {best.ece:.3f}, ")
            f.write(f"Cov90 {best.pi90_coverage:.3f}, p95 {best.inference_time_ms_p95:.1f} ms\n\n")
            
            f.write("## Results\n\n")
            f.write("| Config | Acc | ECE | 90% Cov | Avg Set | Perplexity | p50 ms | p95 ms | Params |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for r in sorted(results, key=lambda x: -x.accuracy):
                f.write(f"| {r.config_name} | {r.accuracy:.3f} | {r.ece:.3f} | ")
                f.write(f"{r.pi90_coverage:.3f} | {r.avg_pred_set_size:.2f} | ")
                f.write(f"{r.perplexity:.1f} | {r.inference_time_ms_p50:.1f} | ")
                f.write(f"{r.inference_time_ms_p95:.1f} | {r.num_parameters:,} |\n")
            
            f.write("\n## VQ Diagnostics\n")
            vq_results = [r for r in results if "vq" in r.config_name.lower()]
            if vq_results:
                f.write("- Average perplexity: {:.1f}\n".format(np.mean([r.perplexity for r in vq_results])))
                f.write("- Best VQ accuracy: {:.3f}\n".format(max(r.accuracy for r in vq_results)))
            
            f.write("\n## Recommendations\n")
            f.write("- Monitor VQ perplexity target: 50-200 (healthy codebook usage)\n")
            f.write("- If perplexity still ~1.0, increase epochs or reduce decay rate further\n")
            f.write("- Check gradient flow through VQ layer if accuracy remains low\n")
        
        print(f"Report saved to {report}")


if __name__ == "__main__":
    print("Ablation Framework with Complete VQ Fixes")
    set_all_seeds(42)
    
    # Load real quadruped dataset
    from pathlib import Path
    data_dir = Path("data")
    
    print("Loading real quadruped data...")
    train_data = torch.load(data_dir / "quadruped_train.pt")
    val_data = torch.load(data_dir / "quadruped_val.pt")
    
    Xtr, ytr = train_data["X"], train_data["y"]
    Xva, yva = val_data["X"], val_data["y"]
    
    print(f"Train: {Xtr.shape}, Val: {Xva.shape}")
    print(f"Classes: {len(torch.unique(ytr))}")
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr, ytr), 
        batch_size=32,  # Smaller batch for more frequent updates
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xva, yva), 
        batch_size=32, 
        shuffle=False
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = f"ablation_vq_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    runner = AblationRunner(train_loader, val_loader, device=device, output_dir=output_dir)
    
    # Run with more epochs for EMA stabilization
    results = runner.run_ablation(epochs=150, seed=123)
    runner.generate_report(results)
    print(f"✅ Done. See {output_dir}/ for JSON, CSV, and report.")