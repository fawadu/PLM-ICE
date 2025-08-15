#!/usr/bin/env python3
"""
Protein Sequence Embedding Generator

This script generates ESM-2 embeddings for protein sequences and saves them as pickle files.
Handles both short sequences and long sequences (>1024 AA) by chunking.
"""

import os
import pickle
import random
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Bio import SeqIO
from sklearn.decomposition import PCA
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             matthews_corrcoef, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split, KFold, cross_val_score

# ESM model imports
import esm
from torch.utils.data import DataLoader, TensorDataset

# Set random seed for reproducibility
random.seed(42)


INPUT_FASTA = "/PATH_TO_FASTA_FILE"
OUTPUT_PKL = "/PATH_TO_PKL_FILE"

def load_fasta(file_path: str) -> List[Tuple[str, str]]:
    """Load sequences from a FASTA file.

    Args:
        file_path: Path to the FASTA file

    Returns:
        List of tuples containing (sequence_id, sequence)
    """
    seqs = SeqIO.parse(file_path, "fasta")
    sequences = list(seqs)
    return [(record.id, str(record.seq)) for record in sequences]


def generate_sequence_embeddings(
        file_path: Optional[str] = None,
        data: Optional[List[Tuple[str, str]]] = None,
        number_of_layers: int = 6,
        visualize: bool = False
) -> List[torch.Tensor]:
    """Generate ESM-2 embeddings for protein sequences.

    Args:
        file_path: Path to FASTA file (alternative to data)
        data: List of (sequence_id, sequence) tuples
        number_of_layers: Which transformer layer to use
        visualize: Whether to show attention maps

    Returns:
        List of embedding tensors for each sequence
    """
    if file_path is not None:
        data = load_fasta(file_path)
    elif data is None:
        raise ValueError("Either file_path or data must be provided")

    # Load ESM model
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    # Convert sequences to tokens
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[number_of_layers], return_contacts=True)

    # Get representations from specified layer
    token_representations = results["representations"][number_of_layers]

    # Average over sequence length (excluding CLS and SEP tokens)
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(
            token_representations[i, 1:tokens_len - 1].mean(0)
        )

    # Visualization option
    if visualize:
        for (_, seq), tokens_len, attention_contacts in zip(data, batch_lens, results["contacts"]):
            plt.matshow(attention_contacts[:tokens_len, :tokens_len])
            plt.title(seq)
            plt.show()

    return sequence_representations


def process_long_sequence(record: Tuple[str, str]) -> torch.Tensor:
    """Process sequences longer than 1024 AA by chunking.

    Args:
        record: Tuple of (sequence_id, sequence)

    Returns:
        Combined embedding tensor
    """
    # Split into chunks of 1024 AA
    chunks = [record[1][i:i + 1024] for i in range(0, len(record[1]), 1024)]
    chunk_data = [(record[0], chunk) for chunk in chunks]

    # Generate embeddings for all chunks
    chunk_embeddings = generate_sequence_embeddings(data=chunk_data)

    # Sum embeddings from chunks
    return sum(chunk_embeddings)


def generate_embeddings(
        file_path: Optional[str] = None,
        data: Optional[List[Tuple[str, str]]] = None
) -> List[torch.Tensor]:
    """Generate embeddings for all sequences, handling both short and long sequences.

    Args:
        file_path: Path to FASTA file (alternative to data)
        data: List of (sequence_id, sequence) tuples

    Returns:
        List of embedding tensors
    """
    sequence_representations = []

    if file_path is not None:
        data = load_fasta(file_path=file_path)
    elif data is None:
        raise ValueError("Either file_path or data must be provided")

    for record in data:
        if len(record[1]) >= 1024:
            # Process long sequences with chunking
            sequence_representations.append(process_long_sequence(record))
            print("Processing long sequence embeddings")
        else:
            # Process short sequences normally
            embeddings = generate_sequence_embeddings(data=[record])
            sequence_representations.extend(embeddings)

    return sequence_representations


def main():
    """Main function to generate and save embeddings."""
    # Generate embeddings for input FASTA
    embeddings = generate_embeddings(file_path=INPUT_FASTA)

    # Prepare data for saving
    training_data = {"embeddings": embeddings}

    # Save embeddings to pickle file
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(training_data, f)

    print(f"Embeddings successfully saved to {OUTPUT_PKL}")


if __name__ == "__main__":
    main()