"""Sequence model: 1D-CNN autoencoder on resampled raw cycles (needs PyTorch).

Each cycle (T_i x 18) is linearly resampled onto L points of normalized time
[0, 1], standardized per channel with fit statistics, and reconstructed by a
small convolutional autoencoder trained on normal cycles only.

Score modes (larger = more anomalous):
- "mean": mean squared error over time and channels
- "peak": 99th percentile over time of the channel-mean squared error after a
  short moving average (keeps brief events that whole-cycle averages dilute)
"""

from __future__ import annotations

import numpy as np

SEQ_LEN = 1024


def resample_cycle(time_s: np.ndarray, signals: np.ndarray, length: int = SEQ_LEN) -> np.ndarray:
    """(T, C) -> (C, length) on normalized time."""
    t = np.asarray(time_s, dtype=np.float64)
    u = (t - t[0]) / (t[-1] - t[0])
    grid = np.linspace(0.0, 1.0, length)
    x = np.asarray(signals, dtype=np.float64)
    return np.stack([np.interp(grid, u, x[:, c]) for c in range(x.shape[1])]).astype(np.float32)


def _build(n_ch: int, width: int):
    import torch.nn as nn

    w = width
    enc = nn.Sequential(
        nn.Conv1d(n_ch, w, 7, stride=2, padding=3),
        nn.GELU(),
        nn.Conv1d(w, 2 * w, 7, stride=2, padding=3),
        nn.GELU(),
        nn.Conv1d(2 * w, 2 * w, 7, stride=2, padding=3),
        nn.GELU(),
        nn.Conv1d(2 * w, w // 2, 5, stride=2, padding=2),  # bottleneck: (w/2) x L/16
    )
    dec = nn.Sequential(
        nn.ConvTranspose1d(w // 2, 2 * w, 4, stride=2, padding=1),
        nn.GELU(),
        nn.ConvTranspose1d(2 * w, 2 * w, 4, stride=2, padding=1),
        nn.GELU(),
        nn.ConvTranspose1d(2 * w, w, 4, stride=2, padding=1),
        nn.GELU(),
        nn.ConvTranspose1d(w, n_ch, 4, stride=2, padding=1),
    )
    return nn.Sequential(enc, dec)


class SeqAutoencoder:
    """Normal-only 1D-CNN autoencoder. fit/score take arrays of shape (N, C, L)."""

    name = "seq_ae"
    input_kind = "sequence"

    def __init__(
        self,
        width: int = 32,
        epochs: int = 80,
        batch_size: int = 32,
        lr: float = 1e-3,
        score_mode: str = "peak",
        smooth: int = 9,
        seed: int = 0,
        device: str | None = None,
    ):
        self.width = width
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.score_mode = score_mode
        self.smooth = smooth
        self.seed = seed
        self.device = device

    # -- pickling: keep weights as numpy, rebuild the module on demand ----------
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_net", None)
        return state

    def _dev(self):
        # CPU by default: deterministic, matches the API runtime, and avoids an intermittent cuDNN crash
        # seen on the dev laptop (torch 2.11+cu128, driver 596.36: illegal memory access on the 2nd fit).
        return self.device or "cpu"

    def _net_on(self):
        import torch

        if getattr(self, "_net", None) is None:
            net = _build(self.n_ch_, self.width)
            net.load_state_dict({k: torch.from_numpy(v) for k, v in self.weights_.items()})
            self._net = net.to(self._dev()).eval()
        return self._net

    def fit(self, S: np.ndarray, feature_names=None) -> SeqAutoencoder:
        import torch

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.n_ch_ = S.shape[1]
        self.mu_ = S.mean(axis=(0, 2), keepdims=True)[0].astype(np.float32)
        self.sd_ = (S.std(axis=(0, 2), keepdims=True)[0] + 1e-6).astype(np.float32)
        dev = self._dev()
        X = torch.from_numpy(((S - self.mu_) / self.sd_).astype(np.float32)).to(dev)
        net = _build(self.n_ch_, self.width).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        self.loss_curve_ = []
        net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(X), generator=g).to(dev)
            tot = 0.0
            for i in range(0, len(X), self.batch_size):
                xb = X[perm[i : i + self.batch_size]]
                loss = torch.mean((net(xb) - xb) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * len(xb)
            sched.step()
            self.loss_curve_.append(tot / len(X))
        self.weights_ = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
        self._net = net.eval()
        return self

    def errors(self, S: np.ndarray) -> np.ndarray:
        """Per-sample squared reconstruction error, shape (N, C, L)."""
        import torch

        net = self._net_on()
        out = []
        with torch.no_grad():
            Xn = ((S - self.mu_) / self.sd_).astype(np.float32)
            for i in range(0, len(Xn), 256):
                xb = torch.from_numpy(Xn[i : i + 256]).to(self._dev())
                out.append(((net(xb) - xb) ** 2).cpu().numpy())
        return np.concatenate(out)

    def score(self, S: np.ndarray) -> np.ndarray:
        e = self.errors(np.asarray(S, dtype=np.float32))
        if self.score_mode == "mean":
            return e.mean(axis=(1, 2))
        per_t = e.mean(axis=1)  # (N, L)
        if self.smooth > 1:
            k = np.ones(self.smooth) / self.smooth
            per_t = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, per_t)
        return np.quantile(per_t, 0.99, axis=1)

    def params(self) -> dict:
        return {
            "width": self.width,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "score_mode": self.score_mode,
            "smooth": self.smooth,
            "seed": self.seed,
            "seq_len": SEQ_LEN,
            "final_train_loss": float(self.loss_curve_[-1]) if getattr(self, "loss_curve_", None) else None,
        }
