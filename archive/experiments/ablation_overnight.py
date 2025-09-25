
"""
Ablation_Framework_Upgraded.py

Committee-ready ablation framework for the Conv2d-VQ-HDP-HSMM model.
Adds:
  - Fixed training budget + deterministic seeds (fairness)
  - Temperature scaling for probability calibration (ECE)
  - Split conformal prediction (classification) with 90% coverage target
  - Dataset-level VQ perplexity aggregation
  - Robust latency (per-sample p50/p95) with CUDA sync
  - Extended AblationResult fields and richer Markdown/CSV reports
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


def set_all_seeds(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TemperatureScaler(nn.Module):
    def __init__(self): 
        super().__init__()
        self.t = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.t.clamp_min(1e-3)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    scaler = TemperatureScaler().to(logits.device)
    nll = nn.CrossEntropyLoss()

    def _closure():
        opt.zero_grad()
        loss = nll(scaler(logits), labels)
        loss.backward()
        return loss

    opt = torch.optim.LBFGS([scaler.t], lr=0.1, max_iter=50, tolerance_grad=1e-7, tolerance_change=1e-9)
    opt.step(_closure)
    return float(scaler.t.detach().item())


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(conf, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += (m.mean()) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def conformal_threshold(probs_cal: np.ndarray, y_cal: np.ndarray, alpha: float = 0.1) -> float:
    nc = 1.0 - probs_cal[np.arange(len(y_cal)), y_cal]
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
    name: str
    use_vq: bool = True
    use_hdp: bool = True
    use_hsmm: bool = True
    use_entropy: bool = True
    use_augmentation: bool = True
    num_vq_codes: int = 256
    commitment_cost: float = 0.4
    hdp_clusters: int = 20
    hsmm_states: int = 10
    description: str = ""


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
    convergence_epoch: int
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
                num_states=config.hsmm_states if config.use_hsmm else 10,
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
            uncertainty = self.entropy(state_seq)
        return {
            "logits": logits,
            "vq_loss": vq_loss,
            "vq_info": vq_info,
            "cluster_probs": cluster_probs,
            "state_seq": state_seq,
            "uncertainty": uncertainty
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AblationRunner:
    def __init__(self, data_loader, val_loader, device: str = "cuda", output_dir: str = "ablation_results"):
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        try:
            from preprocessing.data_augmentation import BehavioralAugmentation
            self.augmenter = BehavioralAugmentation()
        except Exception:
            self.augmenter = None

    def get_ablation_configs(self) -> List[AblationConfig]:
        return [
            AblationConfig("baseline_encoder", False, False, False, False, False, description="Encoder + Classifier only"),
            AblationConfig("vq_only", True, False, False, False, False, description="VQ only"),
            AblationConfig("hdp_only", False, True, False, False, False, description="HDP only"),
            AblationConfig("hsmm_only", False, False, True, False, False, description="HSMM only"),
            AblationConfig("vq_hdp", True, True, False, False, False, description="VQ + HDP"),
            AblationConfig("vq_hsmm", True, False, True, False, False, description="VQ + HSMM"),
            AblationConfig("hdp_hsmm", False, True, True, False, False, description="HDP + HSMM"),
            AblationConfig("vq_hdp_hsmm", True, True, True, False, False, description="VQ+HDP+HSMM (no uncertainty)"),
            AblationConfig("full_no_aug", True, True, True, True, False, description="Full model without augmentation"),
            AblationConfig("full_with_aug", True, True, True, True, True, description="Full model with augmentation"),
            AblationConfig("vq_256_codes_beta_0_4", True, True, True, True, True, num_vq_codes=256, commitment_cost=0.4, description="Recommended VQ fix"),
            AblationConfig("vq_128_codes_beta_0_5", True, True, True, True, True, num_vq_codes=128, commitment_cost=0.5, description="Small codebook"),
            AblationConfig("vq_512_codes_beta_0_25", True, True, True, True, True, num_vq_codes=512, commitment_cost=0.25, description="Large codebook control"),
        ]

    def train_model(self, model: AblationModel, config: AblationConfig, epochs: int = 50) -> Tuple[float, int]:
        model.to(self.device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        best_acc, conv_epoch = 0.0, 0
        max_steps = epochs * len(self.data_loader)
        steps = 0
        for epoch in range(epochs):
            for batch_x, batch_y in self.data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                if config.use_augmentation and self.augmenter is not None:
                    batch_x, batch_y = self.augmenter(batch_x, batch_y, training=True)
                optimizer.zero_grad()
                out = model(batch_x)
                if isinstance(batch_y, tuple):
                    y_a, y_b, lam = batch_y
                    loss = lam * criterion(out["logits"], y_a) + (1 - lam) * criterion(out["logits"], y_b)
                else:
                    loss = criterion(out["logits"], batch_y)
                if "vq_loss" in out and isinstance(out["vq_loss"], dict) and "vq" in out["vq_loss"]:
                    loss = loss + out["vq_loss"]["vq"]
                loss.backward(); optimizer.step()
                steps += 1
                if steps >= max_steps: break
            acc = self.quick_val_accuracy(model)
            if acc > best_acc: best_acc, conv_epoch = acc, epoch
            if steps >= max_steps: break
        return best_acc, conv_epoch

    def quick_val_accuracy(self, model: AblationModel) -> float:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for bx, by in self.val_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                logits = model(bx)["logits"]
                pred = logits.argmax(dim=1)
                total += by.size(0); correct += (pred == by).sum().item()
        return correct / max(total, 1)

    def evaluate_model(self, model: AblationModel, config: AblationConfig) -> AblationResult:
        model.eval()
        all_logits, all_labels = [], []
        all_times_ms = []
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        perplexity_sum, perplexity_batches = 0.0, 0
        with torch.no_grad():
            for bx, by in self.val_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                start = time.perf_counter()
                out = model(bx)
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - start) * 1000.0
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
        # PATCH: make calibration split adjustable & safer for small sets
        CAL_SPLIT = 0.5  # was 0.3
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
        probs_cal = torch.softmax(logits_cal / T, dim=1).cpu().numpy()
        y_cal = labels_cal.cpu().numpy()
        q = conformal_threshold(probs_cal, y_cal, alpha=0.10)
        P = prediction_set_mask(probs_test, q)
        y_test = labels_test.cpu().numpy()
        pi90_coverage = float(P[np.arange(P.shape[0]), y_test].mean()) if P.shape[0] > 0 else 0.0
        avg_set_size = float(P.sum(axis=1).mean()) if P.shape[0] > 0 else 0.0
        p50 = float(np.percentile(all_times_ms, 50)) if all_times_ms else 0.0
        p95 = float(np.percentile(all_times_ms, 95)) if all_times_ms else 0.0
        perplexity = float(perplexity_sum / max(perplexity_batches, 1)) if config.use_vq else 0.0
        memory_mb = (torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0.0
        return AblationResult(
            config_name=config.name,
            accuracy=accuracy,
            loss=avg_loss,
            perplexity=perplexity,
            ece=float(ece),
            inference_time_ms_p50=p50,
            inference_time_ms_p95=p95,
            num_parameters=model.count_parameters(),
            memory_mb=float(memory_mb),
            convergence_epoch=0,
            pi90_coverage=pi90_coverage,
            avg_pred_set_size=avg_set_size
        )

    def run_ablation(self, epochs: int = 50, seed: int = 42) -> Dict[str, AblationResult]:
        configs = self.get_ablation_configs()
        results: Dict[str, AblationResult] = {}
        print("=" * 80); print("Starting Ablation Study"); print("=" * 80)
        for i, cfg in enumerate(configs, 1):
            set_all_seeds(seed)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            print(f"\n[{i}/{len(configs)}] {cfg.name} — {cfg.description}")
            print(f"Components: VQ={cfg.use_vq}, HDP={cfg.use_hdp}, HSMM={cfg.use_hsmm}, Entropy={cfg.use_entropy}, Aug={cfg.use_augmentation}")
            model = AblationModel(cfg).to(self.device)
            print(f"Parameters: {model.count_parameters():,}")
            print("Training (fixed budget)...")
            best_acc, conv_epoch = self.train_model(model, cfg, epochs=epochs)
            print(f"Best quick-val accuracy: {best_acc:.4f} @ epoch {conv_epoch}")
            print("Evaluating...")
            res = self.evaluate_model(model, cfg)
            res.convergence_epoch = conv_epoch
            results[cfg.name] = res
            self.save_results(results)
            print(f"Acc={res.accuracy:.4f}  ECE={res.ece:.4f}  Cov90={res.pi90_coverage:.3f}  Set={res.avg_pred_set_size:.2f}  p95={res.inference_time_ms_p95:.2f}ms  Perplexity={res.perplexity:.1f}")
        return results

    def save_results(self, results: Dict[str, AblationResult]):
        out_json = self.output_dir / "ablation_results.json"
        out_csv  = self.output_dir / "ablation_results.csv"
        with open(out_json, "w") as f:
            json.dump({k: asdict(v) for k, v in results.items()}, f, indent=2)
        headers = ["config_name","accuracy","ece","pi90_coverage","avg_pred_set_size","perplexity","inference_time_ms_p50","inference_time_ms_p95","num_parameters","memory_mb","convergence_epoch","loss"]
        with open(out_csv, "w") as f:
            f.write(",".join(headers) + "\n")
            for _, r in results.items():
                row = [r.config_name, r.accuracy, r.ece, r.pi90_coverage, r.avg_pred_set_size, r.perplexity, r.inference_time_ms_p50, r.inference_time_ms_p95, r.num_parameters, r.memory_mb, r.convergence_epoch, r.loss]
                f.write(",".join(str(x) for x in row) + "\n")
        print(f"Saved: {out_json} and {out_csv}")

    def generate_report(self, results: Dict[str, AblationResult]):
        report = self.output_dir / "ablation_report.md"
        sorted_items = sorted(results.items(), key=lambda kv: kv[1].accuracy, reverse=True)
        best_name, best_res = sorted_items[0]
        def pass_fail(r: AblationResult) -> str:
            ok_acc = (r.accuracy >= 0.85)
            ok_ece = (r.ece <= 0.03)
            ok_cov = (0.88 <= r.pi90_coverage <= 0.92)
            ok_lat = (r.inference_time_ms_p95 <= 100.0)
            return "PASS" if (ok_acc and ok_ece and ok_cov and ok_lat) else "WARN"
        with open(report, "w", encoding="utf-8") as f:
            f.write("# Ablation Study Report\n\n")
            f.write("**Decision line:** Pass if Accuracy ≥ 0.85, ECE ≤ 0.03, 90% coverage ∈ [0.88, 0.92], p95 ≤ 100 ms.\n\n")
            f.write(f"**Best by accuracy**: `{best_name}` — Acc {best_res.accuracy:.3f}, ECE {best_res.ece:.3f}, Cov90 {best_res.pi90_coverage:.3f}, p95 {best_res.inference_time_ms_p95:.1f} ms\n\n")
            f.write("## Results\n\n")
            f.write("| Config | Acc | ECE | 90% Cov | Avg Set | Perplexity | p50 ms | p95 ms | Params |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for name, r in sorted_items:
                f.write(f"| {name} | {r.accuracy:.3f} | {r.ece:.3f} | {r.pi90_coverage:.3f} | {r.avg_pred_set_size:.2f} | {r.perplexity:.1f} | {r.inference_time_ms_p50:.1f} | {r.inference_time_ms_p95:.1f} | {r.num_parameters:,} |\n")
            thr = best_res.accuracy * 0.95
            candidates = [(n, r) for n, r in results.items() if r.accuracy >= thr]
            candidates.sort(key=lambda kv: kv[1].num_parameters)
            if candidates:
                n_min, r_min = candidates[0]
                f.write("\n## Minimal Sufficient Architecture\n")
                f.write(f"- **{n_min}**: Acc {r_min.accuracy:.3f}, Params {r_min.num_parameters:,}, p95 {r_min.inference_time_ms_p95:.1f} ms — **{pass_fail(r_min)}**\n")
            f.write("\n## Recommendations\n")
            f.write("- Prefer configs that pass the decision line; tune those that warn on a single dimension.\n")
            f.write("- If perplexity > 200, reduce codebook size (e.g., 256) or raise commitment cost (β in [0.35, 0.5]).\n")
            f.write("- Recheck calibration after any hyperparameter change (re-fit temperature; recompute ECE/coverage).\n")
        print(f"Report saved to {report}")


if __name__ == "__main__":
    print("Ablation Framework (Upgraded) — Real Quadruped Dataset")
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
        batch_size=64, 
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xva, yva), 
        batch_size=64, 
        shuffle=False
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = f"ablation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    runner = AblationRunner(train_loader, val_loader, device=device, output_dir=output_dir)
    
    # Run full ablation with 100 epochs per model
    results = runner.run_ablation(epochs=100, seed=123)
    runner.generate_report(results)
    print(f"✅ Done. See {output_dir}/ for JSON, CSV, and report.")
