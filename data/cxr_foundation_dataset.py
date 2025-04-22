
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from typing import Dict, Iterable, List, Optional, Tuple, Union

class EmbeddingsDataset(Dataset):
    """Custom dataset for handling embeddings and labels."""
    
    def __init__(self, embeddings: Iterable[np.ndarray], labels: Iterable[int]):
        """Initialize dataset with embeddings and labels.
        
        Args:
            embeddings: Iterable of embedding arrays
            labels: Iterable of label values (0 or 1)
        """
        # Convert to lists if they aren't already
        self.embeddings = [np.asarray(e, dtype=np.float32) for e in embeddings]
        self.labels = list(labels)
        
        # Validate input
        if len(self.embeddings) != len(self.labels):
            raise RuntimeError("Lengths of embeddings and labels must be equal")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        embedding = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return embedding, label


def create_data_loader_from_embeddings(
    embeddings: Iterable[np.ndarray],
    labels: Iterable[int],
    batch_size: int,
    shuffle: bool = True
) -> DataLoader:
    """Create a PyTorch DataLoader from embeddings and labels.
    
    Args:
        embeddings: Iterable of embedding arrays
        labels: Iterable of label values (0 or 1)
        batch_size: Size of batches to return
        shuffle: Whether to shuffle the data
        
    Returns:
        A PyTorch DataLoader containing (embedding, label) pairs
    """
    dataset = EmbeddingsDataset(embeddings, labels)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available()
    )

"""
DIAGNOSIS = "Cardiomegaly"  # Example diagnosis

# Prepare data loaders
train_loader = create_data_loader_from_embeddings(
    embeddings=df_train["embeddings"],
    labels=df_train[diagnosis],
    batch_size=batch_size,
    shuffle=True
)

val_loader = create_data_loader_from_embeddings(
    embeddings=df_validate["embeddings"],
    labels=df_validate[diagnosis],
    batch_size=val_batch_size,
    shuffle=False
)
"""