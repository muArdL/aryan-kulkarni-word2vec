import pickle

import torch
import torch.nn.functional as F

from model import SGNS


EMBED_DIM = 100

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


with open(
    "results/vocab.pkl",
    "rb"
) as f:
    vocab = pickle.load(f)

word_to_idx = vocab["word_to_idx"]
idx_to_word = vocab["idx_to_word"]

vocab_size = len(word_to_idx)

model = SGNS(
    vocab_size,
    EMBED_DIM
).to(DEVICE)

model.load_state_dict(
    torch.load(
        "results/sgns.pt",
        map_location=DEVICE
    )
)

model.eval()

embeddings = (
    model.in_embeddings.weight.data
)

embeddings = F.normalize(
    embeddings,
    p=2,
    dim=1
)


def most_similar(
    word,
    top_k=10
):
    if word not in word_to_idx:
        print(f"\n'{word}' not found in vocabulary.")
        return

    idx = word_to_idx[word]

    query = embeddings[idx]

    similarities = torch.matmul(
        embeddings,
        query
    )

    values, indices = torch.topk(
        similarities,
        top_k + 1
    )

    print(f"\nMost similar to '{word}':")

    count = 0

    for neighbor_idx in indices.tolist():

        if neighbor_idx == idx:
            continue

        print(
            f"{idx_to_word[neighbor_idx]} "
            f"({similarities[neighbor_idx]:.4f})"
        )

        count += 1

        if count == top_k:
            break


def analogy(
    word_a,
    word_b,
    word_c,
    top_k=5
):
    words = [word_a, word_b, word_c]

    for word in words:
        if word not in word_to_idx:
            print(
                f"\n'{word}' not found in vocabulary."
            )
            return

    vec = (
        embeddings[
            word_to_idx[word_b]
        ]
        -
        embeddings[
            word_to_idx[word_a]
        ]
        +
        embeddings[
            word_to_idx[word_c]
        ]
    )

    vec = F.normalize(
        vec.unsqueeze(0),
        p=2,
        dim=1
    ).squeeze(0)

    similarities = torch.matmul(
        embeddings,
        vec
    )

    values, indices = torch.topk(
        similarities,
        top_k + 10
    )

    print(
        f"\n{word_a} : {word_b} :: "
        f"{word_c} : ?"
    )

    excluded = {
        word_to_idx[word_a],
        word_to_idx[word_b],
        word_to_idx[word_c]
    }

    shown = 0

    for idx in indices.tolist():

        if idx in excluded:
            continue

        print(
            f"{idx_to_word[idx]} "
            f"({similarities[idx]:.4f})"
        )

        shown += 1

        if shown == top_k:
            break


if __name__ == "__main__":

    print("\n========== NEAREST WORDS ==========")

    most_similar("king")
    most_similar("man")
    most_similar("woman")
    most_similar("computer")
    most_similar("music")

    print("\n========== ANALOGIES ==========")

    analogy(
        "man",
        "king",
        "woman"
    )

    analogy(
        "paris",
        "france",
        "berlin"
    )

    analogy(
        "boy",
        "man",
        "girl"
    )