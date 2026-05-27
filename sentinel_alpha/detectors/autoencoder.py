"""Denoising autoencoder anomaly detector (PyTorch, CPU).

Trained on rows labeled normal (y==0) within the training fold (semi-supervised
novelty detection). Score = reconstruction MSE.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sentinel_alpha.detectors.base import AnomalyDetector
from sentinel_alpha.config import SEED
from sentinel_alpha.utils.seeding import set_global_seed


class _AENet(nn.Module):
    def __init__(self, n_in: int, bottleneck: int = 8, hidden: int = 32,
                 mid: int = 16, dropout: float = 0.2) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, mid), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mid, bottleneck),
        )
        self.dec = nn.Sequential(
            nn.Linear(bottleneck, mid), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mid, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_in),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(x))


class AEDetector(AnomalyDetector):
    def __init__(self, bottleneck: int = 8, hidden: int = 32, mid: int = 16,
                 dropout: float = 0.2, noise_std: float = 0.05,
                 lr: float = 1e-3, batch_size: int = 64,
                 max_epochs: int = 200, patience: int = 15,
                 val_frac: float = 0.15, random_state: int = SEED) -> None:
        self.bottleneck = bottleneck
        self.hidden = hidden
        self.mid = mid
        self.dropout = dropout
        self.noise_std = noise_std
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_frac = val_frac
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "AEDetector":
        set_global_seed(self.random_state)
        Xf = np.asarray(X, dtype=np.float32)
        if y is not None:
            mask = (np.asarray(y) == 0)
            if mask.sum() >= 64:
                Xf = Xf[mask]
        n = Xf.shape[0]
        n_val = max(1, int(n * self.val_frac))
        # Last segment as validation to mimic temporal split inside the fold.
        Xtr, Xva = Xf[:-n_val], Xf[-n_val:]

        device = "cpu"
        net = _AENet(Xf.shape[1], self.bottleneck, self.hidden, self.mid, self.dropout).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        ds = TensorDataset(torch.from_numpy(Xtr))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(self.random_state))
        x_va = torch.from_numpy(Xva)

        best_val = float("inf")
        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        bad = 0
        for _ in range(self.max_epochs):
            net.train()
            for (xb,) in loader:
                xb_noisy = xb + self.noise_std * torch.randn_like(xb)
                opt.zero_grad()
                pred = net(xb_noisy)
                loss = loss_fn(pred, xb)
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                val_loss = loss_fn(net(x_va), x_va).item()
            if val_loss + 1e-6 < best_val:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break

        net.load_state_dict(best_state)
        net.eval()
        self.net_ = net
        self.best_val_loss_ = best_val
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        self.net_.eval()
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(X, dtype=np.float32))
            rec = self.net_(x).numpy()
        return np.mean((np.asarray(X, dtype=np.float32) - rec) ** 2, axis=1)
