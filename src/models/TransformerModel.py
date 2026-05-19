import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.CustomDatasets import SequentialStockDataset
import mlflow
import mlflow.pytorch


class StockTransformer(nn.Module):
    """
    Hybrid Transformer encoder for time series classification or regression.

    Each timestep in the sequence has two inputs:
        - A raw signal window
        - Handcrafted features summarizing that same window

    Pipeline:
        per timestep:
            raw signal (batch, seq_len, signal_len) → CNN → cnn_features
            handcrafted (batch, seq_len, n_features) ──────→ handcrafted_features
                                                      concat → (batch, seq_len, cnn_out + n_features)
        across timesteps:
            → linear input projection → d_model
            → sinusoidal positional encoding
            → N x TransformerEncoderLayer
            → mean pool
            → linear head → logit

    Args:
        n_features        (int):   Number of handcrafted features per timestep.
        signal_len        (int):   Length of raw signal window per timestep.
        cnn_channels      (list):  Out-channels for each CNN layer
        kernel_size       (int):   CNN kernel size
        d_model           (int):   Internal transformer dimension
        n_heads           (int):   Number of attention heads
        n_layers          (int):   Number of transformer encoder layers
        d_ff              (int):   Feed-forward hidden dimension
        dropout           (float): Dropout rate
        seq_len           (int):   Number of timesteps in sequence
        task              (str):   'classification' or 'regression'. Default: 'classification'
    """

    def __init__(
        self,
        n_features:   int,
        signal_len:   int,
        cnn_channels: list  = None,
        kernel_size:  int   = 3,
        d_model:      int   = 64,
        n_heads:      int   = 4,
        n_layers:     int   = 2,
        d_ff:         int   = 256,
        dropout:      float = 0.1,
        seq_len:      int   = 60,
        task:         str   = "classification",
    ):
        super().__init__()

        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task     = task
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.n_layers = n_layers

        if cnn_channels is None:
            cnn_channels = [16, 32]

        # ── CNN branch (shared across timesteps) ─────────────────────────────
        # Processes raw signal window per timestep.
        cnn_layers = []
        in_ch = 1
        for out_ch in cnn_channels:
            cnn_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # Compute flattened CNN output size after all pooling
        cnn_out_len = signal_len
        for _ in cnn_channels:
            cnn_out_len = cnn_out_len // 2
        self.cnn_flat_size = cnn_channels[-1] * cnn_out_len

        # ── Input projection ─────────────────────────────────────────────────
        # Projects combined (CNN + handcrafted) features into d_model space.
        combined_size = self.cnn_flat_size + n_features
        self.input_projection = nn.Linear(combined_size, d_model)

        # ── Positional encoding ──────────────────────────────────────────────
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len + 10, dropout=dropout)

        # ── Transformer encoder ──────────────────────────────────────────────
        # Each layer: multi-head self-attention → add & norm → FFN → add & norm
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_ff,
            dropout         = dropout,
            batch_first     = True,  # (batch, seq, features)
            norm_first      = True,  # pre-norm: more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ── Head ─────────────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, signals: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signals  (torch.Tensor): Raw signal windows, shape (batch, seq_len, signal_len)
            features (torch.Tensor): Handcrafted features, shape (batch, n_features)

        Returns:
            torch.Tensor: raw logits, shape (batch,)
        """
        batch, seq_len, signal_len = signals.shape

        # ── CNN per timestep ─────────────────────────────────────────────────
        # Fold seq_len into batch so CNN processes each window independently
        # (batch, seq_len, signal_len) → (batch * seq_len, 1, signal_len)
        s = signals.reshape(batch * seq_len, 1, signal_len)
        s = self.cnn(s)                          # (batch * seq_len, cnn_channels[-1], reduced_len)
        s = s.flatten(start_dim=1)               # (batch * seq_len, cnn_flat_size)
        s = s.reshape(batch, seq_len, -1)        # (batch, seq_len, cnn_flat_size)

        # ── Combine CNN + handcrafted features ───────────────────────────────
        x = torch.cat([s, features], dim=-1)     # (batch, seq_len, cnn_flat_size + n_features)

        # ── Transformer ──────────────────────────────────────────────────────
        x = self.input_projection(x)             # (batch, seq_len, d_model)
        x = self.pos_encoding(x)                 # (batch, seq_len, d_model)
        x = self.transformer(x)                  # (batch, seq_len, d_model)
        x = x[:, -1]                       # (batch, d_model) — mean pool
        return self.head(x).squeeze(-1)          # (batch,)

    def fit(
        self,
        train_signals:   np.ndarray,
        train_features:  np.ndarray,
        train_labels:    np.ndarray,
        val_signals:     np.ndarray,
        val_features:    np.ndarray,
        val_labels:      np.ndarray,
        epochs:          int   = 30,
        batch_size:      int   = 64,
        lr:              float = 1e-4,
        weight_decay:    float = 1e-4,
    ) -> dict:
        """
        Train the model with a validation loop each epoch.

        Args:
            train_signals   (np.ndarray): shape (N_train, seq_len, signal_len)
            train_features  (np.ndarray): shape (N_train, n_features)
            train_labels    (np.ndarray) 
            val_signals     (np.ndarray): shape (N_val, seq_len, signal_len)
            val_features    (np.ndarray): shape (N_val, n_features)
            val_labels      (np.ndarray)

        Returns:
            dict with keys 'train_loss', 'train_acc', 'val_loss', 'val_acc'
        """
        train_loader = DataLoader(
            SequentialStockDataset(train_signals, train_features, train_labels),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            SequentialStockDataset(val_signals, val_features, val_labels),
            batch_size=batch_size,
            shuffle=False,
        )

        criterion = nn.BCEWithLogitsLoss() if self.task == "classification" else nn.MSELoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        print("Starting Training Loop")
        with mlflow.start_run() as run:
            mlflow.log_params({
                "epochs":       epochs,
                "batch_size":   batch_size,
                "lr":           lr,
                "weight_decay": weight_decay,
                "task":         self.task,
                "d_model":      self.d_model,
                "n_heads":      self.n_heads,
                "n_layers":     self.n_layers,
            })

            print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>8}")
            print("-" * 55)

            for epoch in range(1, epochs + 1):
                tr_loss, tr_acc = self._run_epoch(train_loader, criterion, optimizer, training=True)
                vl_loss, vl_acc = self._run_epoch(val_loader,   criterion, optimizer, training=False)
                scheduler.step()

                history["train_loss"].append(tr_loss)
                history["train_acc"].append(tr_acc)
                history["val_loss"].append(vl_loss)
                history["val_acc"].append(vl_acc)

                mlflow.log_metrics({
                    "train_loss": tr_loss,
                    "train_acc":  tr_acc,
                    "val_loss":   vl_loss,
                    "val_acc":    vl_acc,
                }, step=epoch)

                print(
                    f"{epoch:>5}  {tr_loss:>10.4f}  {tr_acc:>9.3f}  {vl_loss:>8.4f}  {vl_acc:>8.3f}"
                )

            mlflow.pytorch.log_model(self, "model")
            print(f"Run ID: {run.info.run_id}")
            print(f"Artifact URI: {run.info.artifact_uri}")

        return history

    def evaluate(self, signals: np.ndarray, features: np.ndarray) -> tuple:
        """
        Run inference on sequences.

        Args:
            signals  (np.ndarray): shape (N, seq_len, signal_len)
            features (np.ndarray): shape (N, n_features)

        Returns:
            classification: (preds, probs) as np.ndarrays
            regression:     logits as np.ndarray
        """
        self.eval()
        with torch.no_grad():
            s = torch.tensor(signals,  dtype=torch.float32).to(self.device)
            f = torch.tensor(features, dtype=torch.float32).to(self.device)
            logits = self(s, f)

            if self.task == "classification":
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long()
                return preds.cpu().numpy(), probs.cpu().numpy()
            else:
                return logits.cpu().numpy()

    def _run_epoch(self, loader, criterion, optimizer, training: bool):
        """Single pass over a DataLoader. Returns (avg_loss, accuracy)."""
        self.train() if training else self.eval()

        total_loss, correct, total = 0.0, 0, 0

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for signals, features, labels in loader:
                signals  = signals.to(self.device)
                features = features.to(self.device)
                labels   = labels.to(self.device)

                logits = self(signals, features)         # (batch,)
                loss   = criterion(logits, labels)

                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step()

                total_loss += loss.item() * len(labels)

                if self.task == "classification":
                    preds    = (torch.sigmoid(logits) >= 0.5).long()
                    correct += (preds == labels.long()).sum().item()
                total += len(labels)

        avg_loss = total_loss / total
        accuracy = correct / total if self.task == "classification" else 0.0
        return avg_loss, accuracy

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    Position = which window in the sequence (oldest → most recent).
    """

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                     # (1, max_len, d_model)
        self.register_buffer("pe", pe)           # saved with model, not trained

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)