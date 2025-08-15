
#!/usr/bin/env python3
"""
This script performs k-fold cross-validation and evaluates performance metrics.
"""

import pickle
import random
from typing import List, Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader, TensorDataset

# ========== CONFIGURATION ==========
# Path to embedding files
POSITIVE_EMBEDDINGS_PATH = "PATH_TO_POSITIVE.faa"
NEGATIVE_EMBEDDINGS_PATH = "PATH_TO_NEGATIVE.faa"

# Hyperparameter search space
LEARNING_RATES = [0.00005, 0.00008, 0.0001, 0.0005, 0.001]
BATCH_SIZES = [8, 16, 20, 32, 64]
EPOCHS_LIST = [10, 15, 20, 25]

# Cross-validation parameters
K_FOLDS = 5
TEST_SIZE = 0.2  # Initial train-test split

# Random seed for reproducibility
SEED = 42


# ==================================


class ProteinClassifier(nn.Module):
    """Neural network classifier for protein sequences."""

    def __init__(self, input_size: int):
        super(ProteinClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.fc1.in_features)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_embeddings(file_path: str) -> List[torch.Tensor]:
    """Load embeddings from pickle file."""
    with open(file_path, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data["embeddings"]


def plot_metrics(train_losses: List[float], test_losses: List[float],
                 train_accuracies: List[float], test_accuracies: List[float],
                 epochs: int, fold: int, params: Dict[str, Any]) -> None:
    """Plot training and validation metrics."""
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(range(1, epochs + 1), train_losses, label='Train Loss',
             color='blue', linestyle='-')
    ax1.plot(range(1, epochs + 1), test_losses, label='Test Loss',
             color='blue', linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='center left')

    ax2 = ax1.twinx()
    ax2.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy',
             color='red', linestyle='-')
    ax2.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy',
             color='red', linestyle='--')
    ax2.set_ylabel('Accuracy (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='center right')

    title = (f"Fold {fold} | LR={params['lr']:.5f} | "
             f"Batch={params['batch_size']} | Epochs={params['epochs']}")
    plt.title(title)
    plt.savefig(f"lr_{params['lr']:.5f}_batch_{params['batch_size']}_fold_{fold}.png")
    plt.close()


def run_cv_fold(X_train: torch.Tensor, y_train: torch.Tensor,
                train_idx: np.ndarray, test_idx: np.ndarray,
                input_size: int, params: Dict[str, Any]) -> Dict[str, float]:
    """Run a single fold of cross-validation with given parameters."""
    # Split data for current fold
    x_train, x_test = X_train[train_idx], X_train[test_idx]
    y_train_fold, y_test_fold = y_train[train_idx], y_train[test_idx]

    # Create data loaders
    train_dataset = TensorDataset(x_train, y_train_fold)
    test_dataset = TensorDataset(x_test, y_test_fold)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

    # Initialize model, loss, and optimizer
    model = ProteinClassifier(input_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])

    # Training tracking
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    # Training loop
    for epoch in range(params['epochs']):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for sequences, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(sequences).squeeze(1)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted_train = torch.sigmoid(outputs) > 0.5
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        correct_test = 0
        total_test = 0
        test_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for sequences, labels in test_loader:
                outputs = model(sequences).squeeze(1)
                test_loss += criterion(outputs, labels.float()).item()
                predicted_test = torch.sigmoid(outputs) > 0.5
                total_test += labels.size(0)
                correct_test += (predicted_test == labels).sum().item()
                all_predictions.extend(predicted_test.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate validation metrics
        test_loss = test_loss / len(test_loader)
        test_accuracy = 100 * correct_test / total_test
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch + 1}/{params['epochs']}: "
              f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
              f"Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%")

    # Plot training curves
    plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies,
                 params['epochs'], params['fold'], params)

    # Calculate final metrics
    return {
        'train_accuracy': train_accuracies[-1],
        'test_accuracy': test_accuracies[-1],
        'precision': precision_score(all_labels, all_predictions),
        'recall': recall_score(all_labels, all_predictions),
        'f1': f1_score(all_labels, all_predictions),
        'mcc': matthews_corrcoef(all_labels, all_predictions),
        'auc_roc': roc_auc_score(all_labels, all_predictions),
        'params': params
    }


def main():
    # Set random seeds for reproducibility
    random.seed(SEED)
    torch.manual_seed(SEED)

    # Load and prepare data
    print("Loading data...")
    positive_embeddings = load_embeddings(POSITIVE_EMBEDDINGS_PATH)
    negative_embeddings = load_embeddings(NEGATIVE_EMBEDDINGS_PATH)

    # Create labels (1 for positive, 0 for negative)
    positive_labels = [1] * len(positive_embeddings)
    negative_labels = [0] * len(negative_embeddings)

    # Combine and shuffle datasets
    combined = list(zip(positive_embeddings + negative_embeddings,
                        positive_labels + negative_labels))
    random.shuffle(combined)
    X, y = zip(*combined)

    X = torch.stack(X)
    y = torch.tensor(y)

    # Initial train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, shuffle=True)

    # Cross-validation setup
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    # Store all results for analysis
    all_results = []

    # Hyperparameter search loop
    for lr in LEARNING_RATES:
        for batch_size in BATCH_SIZES:
            for epochs in EPOCHS_LIST:
                print(f"\nTesting Parameters: LR={lr:.5f}, Batch={batch_size}, Epochs={epochs}")

                fold_metrics = {
                    'train_accuracy': [],
                    'test_accuracy': [],
                    'precision': [],
                    'recall': [],
                    'f1': [],
                    'mcc': [],
                    'auc_roc': []
                }

                # Cross-validation loop
                for fold, (train_idx, test_idx) in enumerate(kf.split(X_train)):
                    print(f"\nFold {fold + 1}/{K_FOLDS}")

                    params = {
                        'lr': lr,
                        'batch_size': batch_size,
                        'epochs': epochs,
                        'fold': fold + 1
                    }

                    # Run CV fold
                    results = run_cv_fold(X_train, y_train, train_idx, test_idx,
                                          X_train.shape[1], params)

                    # Store metrics
                    for metric in fold_metrics.keys():
                        fold_metrics[metric].append(results[metric])

                    print(f"Fold {fold + 1} Metrics:")
                    print(f"Precision: {results['precision']:.4f}")
                    print(f"Recall: {results['recall']:.4f}")
                    print(f"F1-score: {results['f1']:.4f}")
                    print(f"MCC: {results['mcc']:.4f}")
                    print(f"AUC-ROC: {results['auc_roc']:.4f}")

                # Calculate mean metrics for this parameter set
                param_results = {
                    'params': {
                        'learning_rate': lr,
                        'batch_size': batch_size,
                        'epochs': epochs
                    },
                    'mean_metrics': {
                        metric: np.mean(values) for metric, values in fold_metrics.items()
                    },
                    'max_metrics': {
                        metric: np.max(values) for metric, values in fold_metrics.items()
                    }
                }

                all_results.append(param_results)

                print(f"\nSummary for LR={lr:.5f}, Batch={batch_size}, Epochs={epochs}:")
                for metric, value in param_results['mean_metrics'].items():
                    print(f"Mean {metric}: {value:.4f}")

    # Find best performing parameter set
    best_result = max(all_results, key=lambda x: x['mean_metrics']['auc_roc'])
    print("\n=== Best Parameter Set ===")
    print(f"Learning Rate: {best_result['params']['learning_rate']:.5f}")
    print(f"Batch Size: {best_result['params']['batch_size']}")
    print(f"Epochs: {best_result['params']['epochs']}")
    print("\nPerformance Metrics:")
    for metric, value in best_result['mean_metrics'].items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()