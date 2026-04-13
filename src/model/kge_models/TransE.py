import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_normal_

from src.utilities.utilities import *

class TransE(nn.Module):
    """
    TransE Knowledge Graph Embedding Model

    Implements the TransE model from "Translating Embeddings for Modeling Multi-relational Data"
    (Bordes et al., 2013) for knowledge graph representation learning.

    Attributes:
        args: Configuration parameters
        kg: Knowledge graph object containing entity and relation information
        ent_embeddings: Entity embedding matrix
        rel_embeddings: Relation embedding matrix
    """

    def __init__(self, args, kg) -> None:
        """
        Initialize TransE model with knowledge graph and configuration parameters

        Args:
            args: Model configuration containing emb_dim, device, margin, neg_ratio
            kg: Knowledge graph object with ent_num and rel_num attributes
        """
        super(TransE, self).__init__()
        self.args = args
        self.kg = kg

        # Initialize embeddings
        self.ent_embeddings = nn.Embedding(self.kg.ent_num, self.args.emb_dim)
        self.rel_embeddings = nn.Embedding(self.kg.rel_num, self.args.emb_dim)

        # Apply Xavier initialization for better convergence
        xavier_normal_(self.ent_embeddings.weight)
        xavier_normal_(self.rel_embeddings.weight)

        # Move to appropriate device
        self.to(self.args.device)

        # Initialize loss function
        self.margin_loss_func = nn.MarginRankingLoss(
            margin=float(self.args.margin),
            reduction="sum"
        )

    def to(self, device):
        """Move model to specified device"""
        super().to(device)
        return self

    def embedding(self) -> tuple:
        """Return entity and relation embeddings"""
        return self.ent_embeddings.weight, self.rel_embeddings.weight

    def ent_norm(self, e: torch.Tensor) -> torch.Tensor:
        """Normalize entity embeddings to unit L2-norm"""
        return F.normalize(e, p=2, dim=-1)

    def rel_norm(self, r: torch.Tensor) -> torch.Tensor:
        """Normalize relation embeddings to unit L2-norm"""
        return F.normalize(r, p=2, dim=-1)

    def score_fun(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Calculate TransE score (L1 distance between h+r and t)

        Lower scores indicate higher plausibility
        """
        h = self.ent_norm(h)
        r = self.rel_norm(r)
        t = self.ent_norm(t)
        return torch.norm(h + r - t, p=1, dim=-1)

    def split_pn_score(self, score: torch.Tensor, label: torch.Tensor) -> tuple:
        """
        Split scores into positive and negative samples

        Args:
            score: Score tensor for all samples
            label: Binary labels (positive = 1, negative = -1)

        Returns:
            Tuple of (positive_scores, negative_scores)
        """
        pos_indices = torch.where(label > 0)
        neg_indices = torch.where(label < 0)

        if pos_indices[0].numel() == 0 or neg_indices[0].numel() == 0:
            raise ValueError("Missing positive or negative samples in batch")

        p_score = score[pos_indices]

        # Reshape negative scores to [num_pos_samples, neg_ratio] and average
        n_score = score[neg_indices].reshape(-1, self.args.neg_ratio)
        n_score = n_score.mean(dim=1)

        return p_score, n_score

    def prepare_indices(self, h, r=None, t=None):
        """
        Process and validate triple indices

        Handles different input formats (separate h,r,t or combined tensor)
        and ensures proper shape and type conversion
        """
        # Case 1: First argument contains the entire triple
        if isinstance(h, torch.Tensor) and len(h.shape) > 1 and h.shape[1] == 3:
            h_idx, r_idx, t_idx = h[:, 0], h[:, 1], h[:, 2]
        # Case 2: Separate head, relation, tail
        elif r is not None and t is not None:
            h_idx, r_idx, t_idx = h, r, t
        # Case 3: First half is heads, second half is relations (for ADVIMP method)
        elif r is None and t is None and self.args.unlearning_method == "ADVIMP":
            n = h.size(0)
            half_n = n // 2
            return h[:half_n], h[half_n:], None
        else:
            raise ValueError("Invalid input format for TransE model")

        # Flatten tensors if needed
        tensors = []
        for idx in [h_idx, r_idx, t_idx]:
            if idx is not None:
                if isinstance(idx, torch.Tensor) and len(idx.shape) > 1:
                    idx = idx.reshape(-1)
                # Ensure LongTensor type
                if not isinstance(idx, (torch.LongTensor, torch.cuda.LongTensor)):
                    idx = idx.long()
            tensors.append(idx)

        return tuple(tensors)

    def margin_loss(self, head, relation, tail, label=None):
        """
        Calculate margin-based ranking loss

        Args:
            head: Head entity indices
            relation: Relation indices
            tail: Tail entity indices
            label: Binary labels (1 for positive, -1 for negative)

        Returns:
            TransE margin loss value
        """
        # Process indices to ensure proper format
        h_idx, r_idx, t_idx = self.prepare_indices(head, relation, tail)

        # Get embeddings
        ent_embeddings, rel_embeddings = self.embedding()

        # Look up embeddings
        h = torch.index_select(ent_embeddings, 0, h_idx)
        r = torch.index_select(rel_embeddings, 0, r_idx)
        t = torch.index_select(ent_embeddings, 0, t_idx)

        # Calculate scores
        score = self.score_fun(h, r, t)

        # Split into positive and negative samples
        p_score, n_score = self.split_pn_score(score, label)

        # Calculate margin loss
        y = torch.tensor([-1], device=self.args.device)
        return self.margin_loss_func(p_score, n_score, y)

    def loss(self, head, relation, tail=None, label=None):
        """
        Calculate normalized loss for a batch

        Args:
            head: Head entity indices
            relation: Relation indices
            tail: Tail entity indices
            label: Binary labels

        Returns:
            Normalized loss value
        """
        batch_size = head.size(0)
        if batch_size == 0:
            return torch.tensor(0.0, device=self.args.device)

        loss = self.margin_loss(head, relation, tail, label) / batch_size
        return loss

    @torch.no_grad()
    def predict(self, head, relation):
        """
        Predict tail entities given head and relation

        Args:
            head: Head entity indices
            relation: Relation indices

        Returns:
            Probability scores for all entities as potential tails
        """
        # Get embeddings
        ent_embeddings, rel_embeddings = self.embedding()

        # Lookup embeddings
        h = torch.index_select(ent_embeddings, 0, head)
        r = torch.index_select(rel_embeddings, 0, relation)

        # Normalize embeddings
        h = self.ent_norm(h)
        r = self.rel_norm(r)
        t_all = self.ent_norm(ent_embeddings)

        # Calculate h + r for prediction
        pred_t = h + r

        # Calculate score against all possible tail entities (using broadcasting)
        # Convert L1 distance to similarity score (9.0 - distance)
        score = 9.0 - torch.norm(pred_t.unsqueeze(1) - t_all, p=1, dim=2)

        # Apply sigmoid to get probability-like scores in [0,1]
        return torch.sigmoid(score)

    def forward(self, head, relation=None):
        """
        Forward pass for the ADVIMP unlearning method

        Args:
            head: Combined tensor where first half contains head entities
                 and second half contains relations
            relation: Not used in this mode (kept for compatibility)

        Returns:
            Prediction scores
        """
        if self.args.unlearning_method != "ADVIMP":
            raise ValueError("Forward method is only designed for ADVIMP unlearning")

        # Split input tensor
        h, r, _ = self.prepare_indices(head, relation)

        # Get embeddings
        ent_embeddings, rel_embeddings = self.embedding()

        # Normalize embeddings
        h = self.ent_norm(h)
        r = self.rel_norm(r)
        t_all = self.ent_norm(ent_embeddings)

        # Calculate scores against all possible tails
        pred_t = h + r
        score = 9.0 - torch.norm(pred_t.unsqueeze(1) - t_all, p=1, dim=2)

        return torch.sigmoid(score)
