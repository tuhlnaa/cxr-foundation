"""
CXR Foundation: Data-Efficient Classifier Training

This script demonstrates how to train a simple neural network classifier using
precomputed ELIXR embeddings (elixr_img_contrastive) from chest X-ray images.
These embeddings are text-aligned image embeddings from the Q-former output in 
ELIXR (https://arxiv.org/abs/2308.01317).

The script assumes precomputed embeddings and labels from the NIH ChestX-ray14 dataset.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Iterable, List, Optional, Tuple, Union
from dataclasses import dataclass
import math

from sklearn.model_selection import train_test_split
from data.cxr_foundation_dataset import create_data_loader_from_embeddings
from data.download_dataset import ELIXRDataLoader


@dataclass
class ModelConfig:
    """Configuration for classifier model."""
    token_num: int = 32
    embeddings_size: int = 128
    learning_rate: float = 0.1
    end_lr_factor: float = 1.0
    dropout: float = 0.0
    decay_steps: int = 1000
    hidden_layer_sizes: List[int] = None
    weight_decay: float = 0.0
    seed: Optional[int] = None


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """Cosine learning rate scheduler with warmup."""
    
    def __init__(self, optimizer, warmup_steps, decay_steps, end_lr_factor=1.0, last_epoch=-1):
        """Initialize cosine scheduler with warmup.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            decay_steps: Total number of decay steps
            end_lr_factor: Final learning rate as a fraction of initial
            last_epoch: The index of the last epoch
        """
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.end_lr_factor = end_lr_factor
        super(CosineWarmupScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / float(self.warmup_steps)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_steps) / float(self.decay_steps - self.warmup_steps)
            progress = min(1.0, progress)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [base_lr * (self.end_lr_factor + (1 - self.end_lr_factor) * cosine_decay) 
                   for base_lr in self.base_lrs]


class ClassifierModel(nn.Module):
    """Neural network classifier for ELIXR embeddings."""
    
    def __init__(self, heads: List[str], config: ModelConfig):
        """Initialize the classifier model.
        
        Args:
            heads: List of output head names (usually diagnosis categories)
            config: ModelConfig instance with model parameters
        """
        super(ClassifierModel, self).__init__()
        
        # Default for hidden layers if not provided
        hidden_layer_sizes = config.hidden_layer_sizes or [512, 256]
        
        # Set seed for reproducibility
        if config.seed is not None:
            torch.manual_seed(config.seed)
        
        # Define model architecture
        self.embedding_size = config.token_num * config.embeddings_size
        self.heads = heads
        
        # Build layers
        layers = []
        
        # Input reshaping and pooling (in forward method)
        input_size = config.embeddings_size  # After pooling
        
        # Hidden layers
        for size in hidden_layer_sizes:
            layers.append(nn.Linear(input_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.Dropout(config.dropout))
            input_size = size
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layer
        self.output_layer = nn.Linear(input_size, len(heads))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, token_num * embedding_size)
            
        Returns:
            Dictionary mapping head names to output predictions
        """
        batch_size = x.shape[0]
        
        # Reshape and pool
        x = x.view(batch_size, -1, self.embedding_size // 32)
        x = torch.mean(x, dim=1)  # Global average pooling
        
        # Apply hidden layers
        x = self.hidden_layers(x)
        
        # Apply output layer
        x = self.output_layer(x)
        x = self.sigmoid(x)
        
        # Create dictionary of outputs
        outputs = {}
        for i, head in enumerate(self.heads):
            outputs[head.lower()] = x[:, i:i+1]
        
        return outputs


class LARSOptimizer(optim.Optimizer):
    """Layer-wise Adaptive Rate Scaling implementation for PyTorch."""
    
    def __init__(self, params, lr=0.01, weight_decay=0.0001, momentum=0.9, eta=0.001):
        """Initialize LARS optimizer.
        
        Args:
            params: Model parameters
            lr: Learning rate
            weight_decay: Weight decay factor
            momentum: Momentum factor
            eta: LARS coefficient
        """
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, eta=eta)
        super(LARSOptimizer, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            lr = group['lr']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                param_state = self.state[p]
                d_p = p.grad.data
                
                weight_norm = torch.norm(p.data)
                grad_norm = torch.norm(d_p)
                
                # Add weight decay
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                
                # Add momentum
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                else:
                    buf = param_state['momentum_buffer']
                
                # Compute local learning rate
                local_lr = eta * weight_norm / (grad_norm + weight_decay * weight_norm)
                
                # Clip local learning rate to global learning rate
                local_lr = min(local_lr, lr)
                
                # Update with momentum
                buf.mul_(momentum).add_(d_p, alpha=local_lr)
                p.data.add_(-buf)
        
        return loss


def binary_metrics(y_true, y_pred, threshold=0.5):
    """Compute binary classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # Apply threshold
    y_pred_binary = (y_pred > threshold).astype(np.float32)
    
    # Calculate metrics
    tp = np.sum((y_true == 1) & (y_pred_binary == 1))
    tn = np.sum((y_true == 0) & (y_pred_binary == 0))
    fp = np.sum((y_true == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true == 1) & (y_pred_binary == 0))
    
    metrics = {
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn
    }
    
    return metrics


def train_model(
    df_train: dict,
    df_validate: dict,
    diagnosis: str,
    config: ModelConfig,
    epochs: int = 100,
    batch_size: int = 512,
    val_batch_size: int = 1,
    device: str = None
) -> ClassifierModel:
    """Train a model on the provided training data.
    
    Args:
        df_train: Dictionary or DataFrame with training data
        df_validate: Dictionary or DataFrame with validation data
        diagnosis: Name of the diagnosis to predict
        config: ModelConfig instance with model parameters
        epochs: Number of training epochs
        batch_size: Batch size for training
        val_batch_size: Batch size for validation
        device: Device to train on ('cuda' or 'cpu')
        
    Returns:
        Trained PyTorch model
    """
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
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
    
    # Create model
    model = ClassifierModel(heads=[diagnosis], config=config).to(device)
    
    # Optimizer with LARS
    optimizer = LARSOptimizer(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = CosineWarmupScheduler(
        optimizer, 
        warmup_steps=int(0.1 * config.decay_steps),  # 10% warmup
        decay_steps=config.decay_steps,
        end_lr_factor=config.end_lr_factor
    )
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for embeddings, labels in train_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(embeddings)
            loss = criterion(outputs[diagnosis.lower()], labels.view(-1, 1))
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Step the scheduler
        scheduler.step()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_metrics = {
            'true_positives': 0,
            'true_negatives': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
        
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings = embeddings.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(embeddings)
                loss = criterion(outputs[diagnosis.lower()], labels.view(-1, 1))
                
                val_loss += loss.item()
                
                # Calculate metrics
                batch_metrics = binary_metrics(
                    labels.view(-1, 1), 
                    outputs[diagnosis.lower()]
                )
                
                for key in val_metrics:
                    val_metrics[key] += batch_metrics[key]
        
        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Metrics - TP: {val_metrics["true_positives"]}, TN: {val_metrics["true_negatives"]}, '
              f'FP: {val_metrics["false_positives"]}, FN: {val_metrics["false_negatives"]}')
    
    return model


def save_model(model: ClassifierModel, path: str):
    """Save the trained model.
    
    Args:
        model: Trained PyTorch model
        path: Path to save the model
    """
    torch.save(model.state_dict(), path)


def load_model(model: ClassifierModel, path: str, device: str = None):
    """Load a trained model.
    
    Args:
        model: PyTorch model architecture
        path: Path to the saved model
        device: Device to load the model to
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device)


def main():
    # Initialize data loader
    data_loader = ELIXRDataLoader()
    
    # Authenticate with HuggingFace
    data_loader.authenticate_huggingface()
    
    # Download data
    file_paths = data_loader.download_data()
    
    # Prepare dataset for specified diagnosis
    # @param ["AIRSPACE_OPACITY", "FRACTURE", "PNEUMOTHORAX", "CONSOLIDATION", "EFFUSION", "PULMONARY_EDEMA", "ATELECTASIS", "CARDIOMEGALY"]
    # @param {type:"slider", min:50, max:400, step:5}
    diagnosis = "FRACTURE"
    max_cases_per_category = 400  # 400 or None
    
    df_labels = data_loader.prepare_dataset(
        file_paths["labels"],
        diagnosis,
        max_cases_per_category
    )

    # @title Separate into training, validation, and testing sets.
    # @param {type:"slider", min:0.05, max:0.8, step:0.05}
    TEST_SPLIT = 0.1 
    df_train, df_validate = train_test_split(df_labels, test_size=TEST_SPLIT)

    print(f"Training set size: {len(df_train)}")
    print(f"Validation set size: {len(df_validate)}")

    # Load embeddings
    df_train_with_embeddings = data_loader.load_embeddings(file_paths["embeddings"], df_train)
    
    print("Data preparation completed.")
    print(f"DataFrame shape: {df_train_with_embeddings.shape}")
    print(df_train_with_embeddings.head(), "\n")

    # Load embeddings
    df_validate_with_embeddings = data_loader.load_embeddings(file_paths["embeddings"], df_validate)
    
    print("Data preparation completed.")
    print(f"DataFrame shape: {df_validate_with_embeddings.shape}")
    print(df_validate_with_embeddings.head(), "\n")



    """Main function to run the training process."""
    # This would typically load data from files or a database
    # For demonstration, this is commented out as we don't have the actual data
    
    # Example usage (commented out as we don't have the actual data)

    # Define constants
    DIAGNOSIS = "FRACTURE"  # Example diagnosis
    
    # Configure model
    config = ModelConfig(
        token_num=32,
        embeddings_size=128,
        learning_rate=0.1,
        dropout=0.0,
        decay_steps=1000,
        hidden_layer_sizes=[512, 256],
        weight_decay=0.0,
        seed=42
    )
    
    # Train model
    model = train_model(
        df_train=df_train_with_embeddings,       # df_train,
        df_validate=df_validate_with_embeddings, # df_validate,
        diagnosis=DIAGNOSIS,
        config=config,
        epochs=100
    )
    
    # Save the model if needed
    save_model(model, 'cxr_classifier_model.pt')



    # # @title Organize the output and display a sample of the predictions
    # val_loader = create_data_loader_from_embeddings(
    #     embeddings=df_validate_with_embeddings["embeddings"],
    #     labels=df_validate_with_embeddings[diagnosis],
    #     batch_size=512,
    #     shuffle=False
    # )
    # rows = []

    # for embeddings, labels in val_loader:
    # #for embeddings, label in val_loader.batch(1):
    #     row = {
    #         f'{DIAGNOSIS}_prediction': model(embeddings)[DIAGNOSIS].numpy().flatten()[0],
    #         f'{DIAGNOSIS}_value': labels.numpy().flatten()[0]
    #     }
    #     rows.append(row)

    # eval_df = pd.DataFrame(rows)
    # eval_df.head()

    # import sklearn
    # import matplotlib.pyplot as plt

    # def plot_curve(x, y, auc, x_label=None, y_label=None, label=None):
    #     fig = plt.figure(figsize=(10, 10))
    #     plt.plot(x, y, label=f'{label} (AUC: %.3f)' % auc, color='black')
    #     plt.legend(loc='lower right', fontsize=18)
    #     plt.xlim([-0.01, 1.01])
    #     plt.ylim([-0.01, 1.01])
    #     if x_label:
    #         plt.xlabel(x_label, fontsize=24)
    #     if y_label:
    #         plt.ylabel(y_label, fontsize=24)
    #     plt.xticks(fontsize=12)
    #     plt.yticks(fontsize=12)
    #     plt.grid(visible=True)


    # labels = eval_df[f'{DIAGNOSIS}_value'].values
    # predictions = eval_df[f'{DIAGNOSIS}_prediction'].values
    # false_positive_rate, true_positive_rate, thresholds = sklearn.metrics.roc_curve(
    #     labels,
    #     predictions,
    #     drop_intermediate=False)
    # auc = sklearn.metrics.roc_auc_score(labels, predictions)
    # plot_curve(false_positive_rate, true_positive_rate, auc, x_label='False Positive Rate', y_label='True Positive Rate', label=DIAGNOSIS)
    # plt.show()

if __name__ == "__main__":
    main()