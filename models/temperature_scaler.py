# temperature_scaler.py
# Minimal temperature scaling to fix high ECE at eval time.
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_T = nn.Parameter(torch.zeros(1))  # T=1 initially

    def forward(self, logits):
        return logits / torch.exp(self.log_T)

    @torch.no_grad()
    def temperature(self):
        return float(torch.exp(self.log_T).item())

    def fit(self, logits, labels, max_iters=1000, lr=0.01):
        self.train()
        opt = optim.LBFGS([self.log_T], lr=lr, max_iter=max_iters)
        def _nll():
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss
        opt.step(_nll)
        self.eval()
        return self.temperature()

def expected_calibration_error(probs, labels, n_bins=15):
    conf, pred = probs.max(dim=1)
    acc = (pred == labels).float()
    ece = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += (acc[mask].mean() - conf[mask].mean()).abs() * mask.float().mean()
    return float(ece.item())
