import collections
import numpy as np
from torch.utils.data import Dataset


class SkipGramDataset(Dataset):
    def __init__(
        self,
        encoded_tokens,
        vocab_size,
        negative_distribution,
        window_size=2,
        negative_samples=5
    ):
        self.tokens = encoded_tokens
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.negative_samples = negative_samples
        self.negative_distribution = negative_distribution

        self.pairs = []

        for center_pos in range(
            window_size,
            len(self.tokens) - window_size
        ):
            center_word = self.tokens[
                center_pos
            ]

            for offset in range(
                -window_size,
                window_size + 1
            ):
                if offset == 0:
                    continue

                context_word = self.tokens[
                    center_pos + offset
                ]

                self.pairs.append(
                    (
                        center_word,
                        context_word
                    )
                )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center_word, context_word = self.pairs[idx]

        negatives = np.random.choice(
            self.vocab_size,
            size=self.negative_samples,
            p=self.negative_distribution
        )

        return (
            center_word,
            context_word,
            negatives
        )


def build_vocab(
    tokens,
    max_vocab_size=10000
):
    counter = collections.Counter(tokens)

    most_common = counter.most_common(
        max_vocab_size - 1
    )

    word_to_idx = {
        "<UNK>": 0
    }

    frequencies = [1]

    for word, count in most_common:
        word_to_idx[word] = len(
            word_to_idx
        )
        frequencies.append(count)

    idx_to_word = {
        idx: word
        for word, idx in word_to_idx.items()
    }

    frequencies = np.array(
        frequencies,
        dtype=np.float64
    )

    negative_distribution = (
        frequencies ** 0.75
    )

    negative_distribution /= (
        negative_distribution.sum()
    )

    return (
        word_to_idx,
        idx_to_word,
        negative_distribution
    )


def encode_tokens(
    tokens,
    word_to_idx
):
    return [
        word_to_idx.get(word, 0)
        for word in tokens
    ]


def build_dataset(
    corpus_path,
    window_size=2,
    max_vocab_size=10000,
    token_limit=300000,
    negative_samples=5
):
    with open(
        corpus_path,
        "r",
        encoding="utf-8"
    ) as f:
        text = f.read()

    tokens = text.split()

    tokens = tokens[:token_limit]

    print(
        f"Tokens used: {len(tokens):,}"
    )

    (
        word_to_idx,
        idx_to_word,
        negative_distribution
    ) = build_vocab(
        tokens,
        max_vocab_size
    )

    encoded_tokens = encode_tokens(
        tokens,
        word_to_idx
    )

    dataset = SkipGramDataset(
        encoded_tokens,
        len(word_to_idx),
        negative_distribution,
        window_size,
        negative_samples
    )

    return (
        dataset,
        word_to_idx,
        idx_to_word
    )