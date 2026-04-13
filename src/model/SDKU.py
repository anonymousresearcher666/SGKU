import torch
import torch.nn as nn

from src.model.SGKU import SGKU


class SDKU(SGKU):
    """Schema-driven Knowledge Unlearning with training-free bias parameters.

    SDKU keeps the base KGE embeddings frozen and learns lightweight bias terms that
    are saved inside the model checkpoint. Biases are applied at scoring time.
    """

    def __init__(self, args, kg, kge_model_class, schema_store=None):
        super().__init__(args=args, kg=kg, kge_model_class=kge_model_class, schema_store=schema_store)

        self._init_sdku_biases()
        self._freeze_kge_parameters()

    def _init_sdku_biases(self) -> None:
        ent_num = int(getattr(self.kg, "ent_num", getattr(self.kg, "n_entity", 0)) or 0)
        if ent_num <= 0:
            ent_num = len(getattr(self.kg, "entity2id", {}))
        rel_num = int(getattr(self.kg, "rel_num", getattr(self.kg, "n_relation", 0)) or 0)
        if rel_num <= 0:
            rel_num = len(getattr(self.kg, "relation2id", {}))

        ent_num = max(0, int(ent_num))
        rel_num = max(0, int(rel_num))

        self.ent_bias = nn.Parameter(torch.zeros((ent_num,), dtype=torch.float))
        self.rel_bias = nn.Parameter(torch.zeros((rel_num,), dtype=torch.float))

        # Reference snapshots for GRPO ratios (kept in sync with refresh_reference_embeddings).
        self.register_buffer("ref_ent_bias", torch.zeros_like(self.ent_bias))
        self.register_buffer("ref_rel_bias", torch.zeros_like(self.rel_bias))

        self.sdku_bias_scale = float(getattr(self.args, "sdku_bias_scale", 1.0))
        self.sdku_bias_clip = getattr(self.args, "sdku_bias_clip", None)

    def _freeze_kge_parameters(self) -> None:
        if bool(getattr(self.args, "sdku_freeze_kge", True)):
            for param in self.kge_model.parameters():
                param.requires_grad = False

    def _refresh_reference_biases(self) -> None:
        if self.ref_ent_bias.shape != self.ent_bias.shape:
            self.ref_ent_bias = torch.zeros_like(self.ent_bias)
        if self.ref_rel_bias.shape != self.rel_bias.shape:
            self.ref_rel_bias = torch.zeros_like(self.rel_bias)
        with torch.no_grad():
            self.ref_ent_bias.copy_(self.ent_bias.detach())
            self.ref_rel_bias.copy_(self.rel_bias.detach())

    @torch.no_grad()
    def refresh_reference_embeddings(self) -> None:
        super().refresh_reference_embeddings()
        self._refresh_reference_biases()

    def _use_reference_bias(self, ent_emb: torch.Tensor, rel_emb: torch.Tensor) -> bool:
        try:
            if ent_emb.data_ptr() == self.ref_ent_embeddings.data_ptr():
                return True
            if rel_emb.data_ptr() == self.ref_rel_embeddings.data_ptr():
                return True
        except Exception:
            return False
        return False

    def _triple_bias(
        self,
        triples: torch.Tensor,
        *,
        use_reference: bool = False,
    ) -> torch.Tensor:
        if triples is None or triples.numel() == 0:
            return torch.zeros((0,), device=triples.device if triples is not None else self.ent_bias.device)

        h = triples[:, 0].long()
        r = triples[:, 1].long()
        t = triples[:, 2].long()

        ent_bias = self.ref_ent_bias if use_reference else self.ent_bias
        rel_bias = self.ref_rel_bias if use_reference else self.rel_bias

        # Clamp indices to avoid accidental out-of-range when timesteps change.
        if ent_bias.numel() > 0:
            h = torch.clamp(h, 0, ent_bias.numel() - 1)
            t = torch.clamp(t, 0, ent_bias.numel() - 1)
        if rel_bias.numel() > 0:
            r = torch.clamp(r, 0, rel_bias.numel() - 1)

        bias = torch.zeros((triples.size(0),), device=triples.device, dtype=torch.float)
        if ent_bias.numel() > 0:
            bias = bias + ent_bias.index_select(0, h) + ent_bias.index_select(0, t)
        if rel_bias.numel() > 0:
            bias = bias + rel_bias.index_select(0, r)

        bias = bias * self.sdku_bias_scale
        clip = self.sdku_bias_clip
        if clip is not None:
            try:
                clip_val = float(clip)
                bias = torch.clamp(bias, min=-clip_val, max=clip_val)
            except (TypeError, ValueError):
                pass
        return bias

    def _triple_logits(self, triples: torch.Tensor, *, ent_emb: torch.Tensor, rel_emb: torch.Tensor) -> torch.Tensor:
        base = super()._triple_logits(triples, ent_emb=ent_emb, rel_emb=rel_emb)
        use_ref = self._use_reference_bias(ent_emb, rel_emb)
        if self.ent_bias.numel() == 0 and self.rel_bias.numel() == 0:
            return base
        bias = self._triple_bias(triples, use_reference=use_ref).to(base.device)
        return base + bias

    def conflict_aware_projection_step(self, *args, **kwargs) -> None:
        # SDKU keeps base embeddings frozen; projection is disabled.
        return

    def predict(self, head: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        """Predict tails with bias-adjusted scores (used in evaluation)."""
        base = self.kge_model.predict(head, relation)
        if self.ent_bias.numel() == 0 and self.rel_bias.numel() == 0:
            return base

        # Build bias matrix: head + relation bias per row, tail bias per column.
        head = head.long()
        relation = relation.long()
        ent_bias = self.ent_bias
        rel_bias = self.rel_bias

        if ent_bias.numel() == 0:
            head_bias = 0.0
            tail_bias = 0.0
        else:
            head = torch.clamp(head, 0, ent_bias.numel() - 1)
            head_bias = ent_bias.index_select(0, head).unsqueeze(1)
            tail_bias = ent_bias.unsqueeze(0)
        if rel_bias.numel() == 0:
            rel_bias_term = 0.0
        else:
            relation = torch.clamp(relation, 0, rel_bias.numel() - 1)
            rel_bias_term = rel_bias.index_select(0, relation).unsqueeze(1)

        bias = head_bias + rel_bias_term
        if isinstance(tail_bias, torch.Tensor):
            bias = bias + tail_bias
        bias = bias * self.sdku_bias_scale
        clip = self.sdku_bias_clip
        if clip is not None:
            try:
                clip_val = float(clip)
                bias = torch.clamp(bias, min=-clip_val, max=clip_val)
            except (TypeError, ValueError):
                pass
        return base + bias
