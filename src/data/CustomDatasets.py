import torch
import numpy as np
from torch.utils.data import Dataset

class HybridDataset(Dataset):
    """
    PyTorch Dataset for (signal, handcrafted features, label) triples.
 
    Args:
        signals   (np.ndarray): Raw signals, shape (N, signal_length)
        features  (np.ndarray): Pre-generated features, shape (N, n_handcrafted)
        labels    (np.ndarray): Binary labels, shape (N,)
    """

    def __init__(self, signals: np.ndarray, features: np.ndarray, labels: np.ndarray):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.signals  = torch.tensor(signals,  dtype=torch.float32).unsqueeze(1).to(device)  # (N, 1, L)
        self.features = torch.tensor(features, dtype=torch.float32).to(device)                # (N, F)
        self.labels   = torch.tensor(labels,   dtype=torch.float32).to(device)                # (N,)
 
    def __len__(self):
        return len(self.labels)
 
    def __getitem__(self, idx):
        return self.signals[idx], self.features[idx], self.labels[idx]
    
class SequentialStockDataset(Dataset):
    """
    PyTorch Dataset for (signal, handcrafted features, label) triples.
 
    Args:
        sequences   (np.ndarray): Raw signals, shape (N, signal_length)
        labels    (np.ndarray): Binary labels, shape (N,)
    """

    def __init__(self, sequences: np.ndarray, signals: np.ndarray, labels: np.ndarray):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequences  = torch.tensor(sequences,  dtype=torch.float32).to(device)  
        self.signals  = torch.tensor(signals,  dtype=torch.float32).to(device) 
        self.labels   = torch.tensor(labels,   dtype=torch.float32).to(device)   
 
    def __len__(self):
        return len(self.labels)
 
    def __getitem__(self, idx):
        return self.sequences[idx], self.signals[idx], self.labels[idx]