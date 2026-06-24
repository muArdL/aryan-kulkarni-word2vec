import torch
import torch.nn as nn
import torch.nn.functional as F


class SGNS(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()

        self.in_embeddings = nn.Embedding(
            vocab_size,
            embed_dim
        )

        self.out_embeddings = nn.Embedding(
            vocab_size,
            embed_dim
        )

        nn.init.uniform_(
            self.in_embeddings.weight,
            -0.5 / embed_dim,
            0.5 / embed_dim
        )

        nn.init.zeros_(
            self.out_embeddings.weight
        )

    def forward(
        self,
        center_words,
        positive_words,
        negative_words
    ):
        center_embeds = self.in_embeddings(
            center_words
        )

        positive_embeds = self.out_embeddings(
            positive_words
        )

        negative_embeds = self.out_embeddings(
            negative_words
        )

        positive_score = torch.sum(
            center_embeds * positive_embeds,
            dim=1
        )

        positive_loss = F.logsigmoid(
            positive_score
        )

        negative_score = torch.bmm(
            negative_embeds,
            center_embeds.unsqueeze(2)
        ).squeeze(2)

        negative_loss = torch.sum(
            F.logsigmoid(
                -negative_score
            ),
            dim=1
        )

        loss = -(positive_loss + negative_loss)

        return loss.mean()