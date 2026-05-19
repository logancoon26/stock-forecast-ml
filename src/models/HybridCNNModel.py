import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import mlflow
import mlflow.pytorch
from src.data.CustomDatasets import HybridDataset
 
 
class HybridCNNModel(nn.Module):
    """
    Hybrid model that combines:
      - A 1D CNN branch for raw signal/sequence input
      - A fully-connected branch for pre-generated (handcrafted) features
    Both branches are concatenated and passed through a classifier head.
 
    Args:
        signal_length (int):    Length of the input 1D signal
        n_handcrafted (int):    Number of pre-generated features
        cnn_channels (list):    Out-channels for each Conv1d layer
        kernel_size (int):      Kernel size for all Conv1d layers
        fc_hidden (int):        Hidden size of the feature branch FC layer
        classifier_hidden (int):Hidden size of the final classifier
        dropout (float):        Dropout rate applied before the classifier
    """
 
    def __init__(
        self,
        signal_length: int,
        n_handcrafted: int,
        cnn_channels: list = None,
        kernel_size: int = 3,
        fc_hidden: int = 64,
        classifier_hidden: int = 128,
        dropout: float = 0.3,
        task: str = "classification"
    ):
        super().__init__()
 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cnn_channels is None:
            cnn_channels = [6, 8]

        # Set Model Task
        self.task = task

        # Set CNN channels for MLFlow tracking
        self.cnn_channels = cnn_channels
 
        # ── CNN branch ──────────────────────────────────────────────────────
        # Input shape: (batch, 1, signal_length)
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
        cnn_out_length = signal_length
        for _ in cnn_channels:
            cnn_out_length = cnn_out_length // 2  # one MaxPool1d(2) per block
        self.cnn_flat_size = cnn_channels[-1] * cnn_out_length
 
        # ── Handcrafted feature branch ───────────────────────────────────────
        # Input shape: (batch, n_handcrafted)
        self.feature_branch = nn.Sequential(
            nn.Linear(n_handcrafted, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, fc_hidden),
            nn.ReLU(),
        )
 
        # ── Classifier head ──────────────────────────────────────────────────
        # Takes concatenated CNN + feature embeddings
        combined_size = self.cnn_flat_size + fc_hidden
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1),  # single logit for binary classification
        )
 
    def forward(self, signal: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signal   (torch.Tensor): Raw signal
            features (torch.Tensor): Hand-crafted features 
 
        Returns:
            torch.Tensor: Raw logits, shape (batch, 1).

        """
        # CNN branch
        x = self.cnn(signal)           # (batch, cnn_channels[-1], reduced_length)
        x = x.flatten(start_dim=1)     # (batch, cnn_flat_size)
 
        # Feature branch
        f = self.feature_branch(features)   # (batch, fc_hidden)
 
        # Combine and classify
        combined = torch.cat([x, f], dim=1)     # (batch, cnn_flat_size + fc_hidden)
        logits = self.classifier(combined)       # (batch, 1)
 
        return logits
    
    def fit(self,
        train_signals: np.ndarray,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        val_signals: np.ndarray,
        val_features: np.ndarray,
        val_labels: np.ndarray,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-3,
    ) -> dict:
        """
        Trains the HybridCNNClassifier with a validation loop each epoch.
    
        Args:
            train_signals   : np.ndarray, shape (N_train, signal_length)
            train_features  : np.ndarray, shape (N_train, n_handcrafted)
            train_labels    : np.ndarray, shape 
            val_signals     : np.ndarray, shape (N_val, signal_length)
            val_features    : np.ndarray, shape (N_val, n_handcrafted)
            val_labels      : np.ndarray, shape 
            epochs          : Number of training epochs
            batch_size      : Samples per batch
            lr              : Adam learning rate
            device          : 'cuda', 'cpu', or None 
    
        Returns:
            dict with keys 'train_loss', 'train_acc', 'val_loss', 'val_acc'
            — each a list of floats, one entry per epoch.
    
        """
        train_loader = DataLoader(
            HybridDataset(train_signals, train_features, train_labels),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            HybridDataset(val_signals, val_features, val_labels),
            batch_size=batch_size,
            shuffle=False,
        )
    
        criterion = nn.BCEWithLogitsLoss() if self.task == "classification" else nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
        with mlflow.start_run(run_name="second_test"):
            # Log hyperparameters
            mlflow.log_params({
                "epochs":           epochs,
                "batch_size":       batch_size,
                "lr":               lr,
                "task":             self.task,
                "cnn_channels":     self.cnn_channels
            })

            for epoch in range(1, epochs + 1):
                tr_loss, tr_acc = self._run_epoch(train_loader, criterion, optimizer, self.device, training=True)
                vl_loss, vl_acc = self._run_epoch(val_loader,   criterion, optimizer, self.device, training=False)
        
                history["train_loss"].append(tr_loss)
                history["train_acc"].append(tr_acc)
                history["val_loss"].append(vl_loss)
                history["val_acc"].append(vl_acc)
        
                print(
                    f"Epoch {epoch:>3}/{epochs} | "
                    f"Train loss: {tr_loss:.4f}  acc: {tr_acc:.3f} | "
                    f"Val loss: {vl_loss:.4f}  acc: {vl_acc:.3f}"
                )

                mlflow.log_metrics({
                    "train_loss": tr_loss,
                    "train_acc":  tr_acc,
                    "val_loss":   vl_loss,
                    "val_acc":    vl_acc,
                }, step=epoch)

            # Log final model weights
            mlflow.pytorch.log_model(self, "model")
    
        return history
    
    def evaluate(self, signals: np.ndarray, features: np.ndarray) -> tuple:
        self.eval()
        with torch.no_grad():
            signal_tensor  = torch.tensor(signals,  dtype=torch.float32).unsqueeze(1).to(self.device)
            feature_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)

            logits = self(signal_tensor, feature_tensor).squeeze(1)
            if self.task == "classification":
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long()
                return preds.cpu().numpy(), probs.cpu().numpy()
            else:
                return logits.cpu().numpy()
    
    def _run_epoch(self, loader, criterion, optimizer, device, training: bool):
        """Single pass over a DataLoader. Returns (avg_loss, accuracy)."""
        self.train() if training else self.eval()
    
        total_loss, correct, total = 0.0, 0, 0
    
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for signals, features, labels in loader:
                signals  = signals.to(device)
                features = features.to(device)
                labels   = labels.to(device)
    
                logits = self(signals, features).squeeze(1)   # (batch,)
                loss   = criterion(logits, labels)
    
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
    
                total_loss += loss.item() * len(labels)
                if self.task == "classification":
                    preds    = (torch.sigmoid(logits) >= 0.5).long()
                    correct += (preds == labels.long()).sum().item()
                else:
                    correct += 0
                total   += len(labels)
    
        return total_loss / total, correct / total