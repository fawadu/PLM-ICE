#!/usr/bin/env python3
"""
This script loads a trained model and tests it on sequence embeddings.
"""

import pickle
import torch
import torch.nn as nn
from Bio import SeqIO

# Path to the trained model
MODEL_PATH = '/PATH/TO/inp_model.txt'  # Update this path

# Path to the test embeddings file (.pkl)
TEST_EMBEDDINGS_FILE = '/PATH/TO/test_embeddings.pkl'  # Update this path

# Optional: Path to the test FASTA file
TEST_FASTA_FILE = None  # Set to None or provide path like '/PATH/TO/test.fasta'

# Optional: Path to save predicted sequences
OUTPUT_PREDICTED_FILE = None  # Set to None or provide path like '/PATH/TO/predictions.fasta'

# =========================================

class ProteinClassifier(nn.Module):
    """Neural network classifier for protein sequences."""

    def __init__(self, input_size):
        super(ProteinClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = x.view(-1, self.fc1.in_features)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    # Load the test embeddings
    with open(TEST_EMBEDDINGS_FILE, 'rb') as file:
        loaded_file = pickle.load(file)

    blast_hits = loaded_file["embeddings"]
    blast_hits = torch.stack(blast_hits)

    # Initialize and load model
    input_size = blast_hits.shape[1]
    model = ProteinClassifier(input_size=input_size)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()  # Set model to evaluation mode

    # Make predictions
    blast_hits_test = []
    for i in blast_hits:
        with torch.no_grad():
            output = model(i)
            output = output.squeeze(1)
            predicted_label = torch.sigmoid(output) > 0.5
            blast_hits_test.append(predicted_label)

    # Count predictions
    refneg = sum(1 for i in blast_hits_test if i == 0)
    inp = sum(1 for i in blast_hits_test if i == 1)

    # Handle FASTA file if provided
    if TEST_FASTA_FILE:
        try:
            test_seqs = list(SeqIO.parse(TEST_FASTA_FILE, 'fasta'))
            indexes_to_pull = [index for index, value in enumerate(blast_hits_test) if value == 1]
            pulled_seqs = [test_seqs[i] for i in indexes_to_pull]

            if OUTPUT_PREDICTED_FILE:
                SeqIO.write(pulled_seqs, handle=OUTPUT_PREDICTED_FILE, format='fasta')
                print(f"Predicted sequences written to {OUTPUT_PREDICTED_FILE}")

            print(f"Number of predicted sequences: {len(pulled_seqs)}")
        except FileNotFoundError:
            print(f"Warning: FASTA file not found at {TEST_FASTA_FILE}")

    # Print summary
    print(f"""
    Sequences predicted as class INP: {inp}
    Sequences predicted as class RefNeg: {refneg}
    Total Number of sequences: {len(blast_hits_test)}
    """)


if __name__ == "__main__":
    main()