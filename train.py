import os
import pickle
import random
import numpy as np

import torch
from torch.utils.data import DataLoader

from dataset import build_dataset
from model import SGNS


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


EMBED_DIM = 100
WINDOW_SIZE = 2
NEGATIVE_SAMPLES = 5

BATCH_SIZE = 1024
EPOCHS = 2
LEARNING_RATE = 0.003

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def train():
    dataset, word_to_idx, idx_to_word = (
        build_dataset(
            "data/text8",
            window_size=WINDOW_SIZE,
            max_vocab_size=10000,
            token_limit=300000,
            negative_samples=NEGATIVE_SAMPLES
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    vocab_size = len(word_to_idx)

    model = SGNS(
        vocab_size,
        EMBED_DIM
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0

        for (
            center_words,
            positive_words,
            negative_words
        ) in loader:

            center_words = center_words.to(
                DEVICE
            )

            positive_words = positive_words.to(
                DEVICE
            )

            negative_words = negative_words.to(
                DEVICE
            )

            optimizer.zero_grad()

            loss = model(
                center_words,
                positive_words,
                negative_words
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = (
            total_loss / len(loader)
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"Loss: {avg_loss:.4f}"
        )

    os.makedirs(
        "results",
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        "results/sgns.pt"
    )

    with open(
        "results/vocab.pkl",
        "wb"
    ) as f:
        pickle.dump(
            {
                "word_to_idx": word_to_idx,
                "idx_to_word": idx_to_word
            },
            f
        )

    print(
        "\nTraining complete."
    )


if __name__ == "__main__":
    train()