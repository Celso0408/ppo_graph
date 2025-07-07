import torch
import torch.nn as nn

class TripleScoringModel(nn.Module):
    """Neural network for scoring knowledge graph triples."""

    def __init__(self, vocab_size: int, predicate_size: int, embedding_dim: int = 32) -> None:
        """Construct the model.

        Parameters
        ----------
        vocab_size:
            Size of the entity vocabulary.
        predicate_size:
            Number of unique predicates.
        embedding_dim:
            Dimensionality of the entity and predicate embeddings.
        """
        super().__init__()
        self.entity_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.predicate_embedding = nn.Embedding(predicate_size, embedding_dim)

        # ⬇️ Scoring layer: [s | p | o] → score
        self.fc = nn.Linear(embedding_dim * 3, 1)

    def forward(self, triple_ids: torch.Tensor) -> torch.Tensor:
        """Score a batch of triples.

        Parameters
        ----------
        triple_ids:
            Tensor of shape ``(batch_size, 3)`` containing subject,
            predicate and object token IDs. A 1-D tensor of length 3 is
            also accepted and will be treated as a batch of size one.

        Returns
        -------
        torch.Tensor
            Tensor of scores with shape ``(batch_size,)``.
        """

        # Ensure the input is 2D: [batch_size, 3]
        if triple_ids.ndim == 1:
            triple_ids = triple_ids.unsqueeze(0)

        subj = triple_ids[:, 0]
        pred = triple_ids[:, 1]
        obj = triple_ids[:, 2]

        s_emb = self.entity_embedding(subj)
        p_emb = self.predicate_embedding(pred)
        o_emb = self.entity_embedding(obj)

        triple_repr = torch.cat([s_emb, p_emb, o_emb], dim=1)
        return self.fc(triple_repr).squeeze()

