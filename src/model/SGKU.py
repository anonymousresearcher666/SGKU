import torch
import torch.nn as nn
from src.utilities.utilities import *


class SGKU(nn.Module):
    """Schema-guided Knowledge Unlearning for knowledge graphs."""

    def __init__(self, args, kg, kge_model_class, schema_store=None):
        """
        Initialize SGKU model.

        Args:
            args: Arguments with model configuration
            kg: Knowledge graph instance
            schema_store: Optional schema store for pattern retrieval
        """
        # Initialize
        super(SGKU, self).__init__()

        # Initialize the base KGE model
        self.kge_model_class = kge_model_class  # Assign this BEFORE using it
        self.kge_model = self.kge_model_class(args, kg)

        self.args = args
        self.kg = kg
        self.schema_store = schema_store
        self.huber_loss = nn.HuberLoss(reduction="sum")
        self.entity_distill_mask = None

        # Tracking variables
        self._last_warning_time = 0
        self._loss_counter = 0
        self.kge_model_class = kge_model_class

        # Configure hyperparameters
        self._set_hyperparameters()
        self._init_paper_state()

    def _set_hyperparameters(self):
        """Set GRPO hyperparameters with defaults."""
        # Paper-aligned GRPO hyperparameters (keep legacy names as fallbacks).
        self.args.epsilon_clip = getattr(self.args, "epsilon_clip", getattr(self.args, "epsilon_grpo", 0.2))
        self.args.lambda_kl = getattr(self.args, "lambda_kl", getattr(self.args, "beta_grpo", 0.001))
        self.args.grpo_lambda = getattr(self.args, "grpo_lambda", 1.0)
        self.args.lambda_boundary = getattr(self.args, "lambda_boundary", getattr(self.args, "boundary_lambda", 0.5))
        self.args.lambda_neg = getattr(self.args, "lambda_neg", getattr(self.args, "lambda_neg_sampling", 0.1))

        self.args.reference_refresh_m = getattr(self.args, "reference_refresh_m", 10)
        self.args.tau_kl = getattr(self.args, "tau_kl", 0.03)
        self.args.kl_refresh_ema_alpha = getattr(self.args, "kl_refresh_ema_alpha", 0.1)
        self.args.projection_every_k = getattr(self.args, "projection_every_k", 20)
        self.args.projection_top_n = getattr(self.args, "projection_top_n", 10)
        self.args.projection_lambda = getattr(self.args, "projection_lambda", getattr(self.args, "gradient_projection_weight", 0.5))

        self.args.temperature_min = getattr(self.args, "temperature_min", 0.1)
        self.args.score_ema_alpha = getattr(self.args, "score_ema_alpha", 0.05)
        # Boundary preservation can be expensive if boundary entity set is large.
        # Sample a subset each step to keep runtime bounded.
        self.args.boundary_batch_size = getattr(self.args, "boundary_batch_size", 4096)

        # Grouping strategy (paper default: relation).
        self.args.grouping_strategy = getattr(self.args, 'grouping_strategy', 'relation')

    def _init_paper_state(self) -> None:
        # EMA statistics for score standardization (Eq. standardization).
        self.register_buffer("score_ema_mu", torch.tensor(0.0))
        self.register_buffer("score_ema_sigma", torch.tensor(1.0))
        self.register_buffer("score_ema_initialized", torch.tensor(False))

        # Reference (theta_old) and initial (theta_init) embeddings.
        # IMPORTANT: buffers must have correct shapes so checkpoints can be loaded on MPS/CPU without size-mismatch.
        ent_w = self.kge_model.ent_embeddings.weight.detach()
        rel_w = self.kge_model.rel_embeddings.weight.detach()
        self.register_buffer("ref_ent_embeddings", torch.zeros_like(ent_w))
        self.register_buffer("ref_rel_embeddings", torch.zeros_like(rel_w))
        self.register_buffer("init_ent_embeddings", torch.zeros_like(ent_w))
        self.register_buffer("init_rel_embeddings", torch.zeros_like(rel_w))

        self.boundary_entities = torch.empty((0,), dtype=torch.long)
        self.optimizer = getattr(self, "optimizer", None)
        self.group_baseline_pos = None
        self.group_baseline_neg = None
        self.group_baseline_initialized = False
        self.last_batch_kl = None

    def save_embeddings(self):
        """Save current embeddings as reference for GRPO and distillation."""
        # Legacy entrypoint used elsewhere in the codebase.
        self.refresh_reference_embeddings()

    def embedding(self):
        """Get current embeddings.

        Returns:
            Tuple of (entity_embeddings, relation_embeddings)
        """
        return self.kge_model.ent_embeddings.weight, self.kge_model.rel_embeddings.weight

    def predict(self, head, relation):
        """Delegate prediction to the underlying KGE model."""
        return self.kge_model.predict(head, relation)

    def old_embeddings(self):
        """Return reference (theta_old) embeddings used for GRPO ratios/rewards."""
        if self.ref_ent_embeddings.numel() == 0:
            self.refresh_reference_embeddings()
        return self.ref_ent_embeddings, self.ref_rel_embeddings

    @torch.no_grad()
    def save_initial_embeddings(self) -> None:
        """Freeze theta_init for boundary preservation."""
        ent = self.kge_model.ent_embeddings.weight.detach()
        rel = self.kge_model.rel_embeddings.weight.detach()
        if self.init_ent_embeddings.numel() == 0 or self.init_ent_embeddings.shape != ent.shape:
            self.init_ent_embeddings = ent.clone()
        else:
            self.init_ent_embeddings.copy_(ent)
        if self.init_rel_embeddings.numel() == 0 or self.init_rel_embeddings.shape != rel.shape:
            self.init_rel_embeddings = rel.clone()
        else:
            self.init_rel_embeddings.copy_(rel)

    @torch.no_grad()
    def refresh_reference_embeddings(self) -> None:
        """Refresh theta_old snapshot (paper: every M steps)."""
        ent = self.kge_model.ent_embeddings.weight.detach()
        rel = self.kge_model.rel_embeddings.weight.detach()
        if self.ref_ent_embeddings.numel() == 0 or self.ref_ent_embeddings.shape != ent.shape:
            self.ref_ent_embeddings = ent.clone()
        else:
            self.ref_ent_embeddings.copy_(ent)
        if self.ref_rel_embeddings.numel() == 0 or self.ref_rel_embeddings.shape != rel.shape:
            self.ref_rel_embeddings = rel.clone()
        else:
            self.ref_rel_embeddings.copy_(rel)

    def set_boundary_entities(self, boundary_entities: torch.Tensor) -> None:
        if boundary_entities is None:
            self.boundary_entities = torch.empty((0,), dtype=torch.long, device=self.args.device)
            return
        if not torch.is_tensor(boundary_entities):
            boundary_entities = torch.tensor(boundary_entities, dtype=torch.long)
        self.boundary_entities = boundary_entities.to(self.args.device)

    def _is_distance_based(self) -> bool:
        kge = str(getattr(self.args, "kge", "")).lower()
        return kge in {"transe", "rotate"}

    def _scores_to_logits(self, scores: torch.Tensor) -> torch.Tensor:
        # Paper standardizes a "higher-is-better" logit across KGE families.
        return -scores if self._is_distance_based() else scores

    def _update_and_standardize(self, raw_logits: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        alpha = float(getattr(self.args, "score_ema_alpha", 0.05))
        alpha = max(0.0, min(1.0, alpha))
        with torch.no_grad():
            batch_mu = raw_logits.detach().mean()
            batch_sigma = raw_logits.detach().std(unbiased=False)
            batch_sigma = torch.clamp(batch_sigma, min=eps)
            if not bool(self.score_ema_initialized.item()):
                self.score_ema_mu.copy_(batch_mu)
                self.score_ema_sigma.copy_(batch_sigma)
                self.score_ema_initialized.copy_(torch.tensor(True, device=self.score_ema_initialized.device))
            else:
                self.score_ema_mu.mul_(1.0 - alpha).add_(alpha * batch_mu)
                self.score_ema_sigma.mul_(1.0 - alpha).add_(alpha * batch_sigma)
        return (raw_logits - self.score_ema_mu) / (self.score_ema_sigma + eps)

    def _triple_logits(self, triples: torch.Tensor, *, ent_emb: torch.Tensor, rel_emb: torch.Tensor) -> torch.Tensor:
        h, r, t = self._get_embeddings_batch(ent_emb, rel_emb, triples)
        scores = self.kge_model.score_fun(h, r, t)
        return self._scores_to_logits(scores)

    def _group_weights_for_ids(self, group_ids: torch.Tensor, group_weight_tensor: torch.Tensor) -> torch.Tensor:
        if group_weight_tensor is None or group_weight_tensor.numel() == 0:
            return torch.ones((group_ids.size(0),), device=group_ids.device)
        group_ids = group_ids.long()
        max_gid = int(group_weight_tensor.size(0) - 1)
        if max_gid < 0:
            return torch.ones((group_ids.size(0),), device=group_ids.device)
        group_ids = torch.clamp(group_ids, min=0, max=max_gid)
        return group_weight_tensor.to(group_ids.device)[group_ids]

    def _group_indices_by_relation(self, triples: torch.Tensor) -> dict:
        groups = {}
        rels = triples[:, 1].detach().cpu().tolist()
        for idx, rel in enumerate(rels):
            groups.setdefault(int(rel), []).append(idx)
        return groups

    def paper_total_loss(
        self,
        *,
        retain_triples: torch.Tensor,
        forget_triples: torch.Tensor,
        group_ids_retain: torch.Tensor = None,
        group_ids_forget: torch.Tensor = None,
        group_weight_tensor: torch.Tensor = None,
    ) -> torch.Tensor:
        """Paper-aligned SGKU objective (GRPO + KL + boundary + optional retain-negative loss)."""
        if retain_triples is None or forget_triples is None:
            return torch.tensor(0.0, device=self.args.device)
        if retain_triples.numel() == 0 and forget_triples.numel() == 0:
            return torch.tensor(0.0, device=self.args.device)

        device = self.args.device
        retain_triples = retain_triples.to(device)
        forget_triples = forget_triples.to(device)
        if group_ids_retain is None:
            group_ids_retain = retain_triples[:, 1].long()
        if group_ids_forget is None:
            group_ids_forget = forget_triples[:, 1].long()
        group_ids_retain = group_ids_retain.to(device)
        group_ids_forget = group_ids_forget.to(device)

        ent, rel = self.kge_model.embedding()
        ref_ent, ref_rel = self.old_embeddings()

        # Build unified batch B (paper: group-balanced sampling is approximated by 50/50 loaders).
        batch_triples = torch.cat([retain_triples, forget_triples], dim=0)
        batch_group_ids = torch.cat([group_ids_retain, group_ids_forget], dim=0).long()
        labels = torch.cat(
            [
                torch.ones((retain_triples.size(0),), device=device),
                -torch.ones((forget_triples.size(0),), device=device),
            ],
            dim=0,
        )

        raw_logits = self._triple_logits(batch_triples, ent_emb=ent, rel_emb=rel)
        raw_logits_ref = self._triple_logits(batch_triples, ent_emb=ref_ent, rel_emb=ref_rel)

        # Standardize with EMA (Eq. standardization).
        z = self._update_and_standardize(raw_logits)
        # Reference logits standardized with the same EMA (keeps ratios well-scaled).
        z_ref = (raw_logits_ref - self.score_ema_mu) / (self.score_ema_sigma + 1e-8)

        # Group policies p_theta^g via softmax within each semantic group.
        # Important for MPS performance: avoid per-triple Python loops and repeated CPU↔device sync.
        max_gid = int(batch_group_ids.max().item()) if batch_group_ids.numel() > 0 else 0
        group_num = int(group_weight_tensor.numel() or 0) if group_weight_tensor is not None else 0
        if group_num <= max_gid:
            # Weight tensor is missing/undersized; fall back to uniform weights.
            group_weight_tensor = None
            group_num = max_gid + 1
        if group_num <= 0:
            group_num = 1
        batch_group_ids = torch.clamp(batch_group_ids, min=0, max=group_num - 1)

        # Per-group sizes (on-device).
        ones = torch.ones_like(batch_group_ids, dtype=torch.float)
        counts = torch.zeros((group_num,), device=device, dtype=torch.float).scatter_add_(0, batch_group_ids, ones)

        t_min = float(getattr(self.args, "temperature_min", 0.1))
        t_min = max(1e-4, t_min)
        temp_per_group = torch.maximum(
            torch.full((group_num,), t_min, device=device, dtype=torch.float),
            1.0 / torch.clamp(counts, min=1.0),
        )

        p = torch.zeros_like(z)
        p_ref = torch.zeros_like(z_ref)
        # Loop only over active groups (<= batch size), not over batch elements.
        active_groups = torch.where(counts > 0)[0].detach().cpu().tolist()
        for gid in active_groups:
            idx = torch.where(batch_group_ids == int(gid))[0]
            if idx.numel() <= 0:
                continue
            temp = temp_per_group[int(gid)]
            v = z.index_select(0, idx) / temp
            v = v - v.max()
            p[idx] = torch.softmax(v, dim=0)

            v_ref = z_ref.index_select(0, idx) / temp
            v_ref = v_ref - v_ref.max()
            p_ref[idx] = torch.softmax(v_ref, dim=0)

        eps = 1e-8
        p = torch.clamp(p, min=eps)
        p_ref = torch.clamp(p_ref, min=eps)

        # Rewards from reference model (paper Eq. reward), signed by label.
        rewards = torch.sigmoid(z_ref) * labels

        # Per-group, per-label baselines b_g^+, b_g^- (paper Eq. advantage).
        # Vectorized on-device with scatter_add to avoid Python loops.
        pos_mask = (labels > 0).to(rewards.dtype)
        neg_mask = (labels < 0).to(rewards.dtype)

        pos_sum = torch.zeros((group_num,), device=device, dtype=rewards.dtype).scatter_add_(0, batch_group_ids, rewards * pos_mask)
        pos_cnt = torch.zeros((group_num,), device=device, dtype=rewards.dtype).scatter_add_(0, batch_group_ids, pos_mask)
        neg_sum = torch.zeros((group_num,), device=device, dtype=rewards.dtype).scatter_add_(0, batch_group_ids, rewards * neg_mask)
        neg_cnt = torch.zeros((group_num,), device=device, dtype=rewards.dtype).scatter_add_(0, batch_group_ids, neg_mask)
        pos_mean = pos_sum / torch.clamp(pos_cnt, min=1.0)
        neg_mean = neg_sum / torch.clamp(neg_cnt, min=1.0)

        # EMA of per-group baselines to reduce minibatch variance.
        ema_beta = float(getattr(self.args, "group_baseline_ema", 0.1))
        ema_beta = max(0.0, min(1.0, ema_beta))
        if self.group_baseline_pos is None or self.group_baseline_neg is None:
            self.group_baseline_pos = pos_mean.detach().clone()
            self.group_baseline_neg = neg_mean.detach().clone()
            self.group_baseline_initialized = True
        elif self.group_baseline_pos.numel() != group_num:
            self.group_baseline_pos = pos_mean.detach().clone()
            self.group_baseline_neg = neg_mean.detach().clone()
            self.group_baseline_initialized = True
        else:
            with torch.no_grad():
                pos_mask_groups = pos_cnt > 0
                neg_mask_groups = neg_cnt > 0
                if bool(pos_mask_groups.any()):
                    self.group_baseline_pos[pos_mask_groups] = (
                        (1.0 - ema_beta) * self.group_baseline_pos[pos_mask_groups]
                        + ema_beta * pos_mean.detach()[pos_mask_groups]
                    )
                if bool(neg_mask_groups.any()):
                    self.group_baseline_neg[neg_mask_groups] = (
                        (1.0 - ema_beta) * self.group_baseline_neg[neg_mask_groups]
                        + ema_beta * neg_mean.detach()[neg_mask_groups]
                    )

        baseline_pos = self.group_baseline_pos if self.group_baseline_initialized else pos_mean
        baseline_neg = self.group_baseline_neg if self.group_baseline_initialized else neg_mean
        baselines = torch.where(
            labels > 0,
            baseline_pos.index_select(0, batch_group_ids),
            baseline_neg.index_select(0, batch_group_ids),
        )

        advantages = rewards - baselines

        ratios = p / p_ref
        clip_eps = float(getattr(self.args, "epsilon_clip", 0.2))
        clipped = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps)
        l_clip = torch.minimum(ratios * advantages, clipped * advantages)

        weights = self._group_weights_for_ids(batch_group_ids, group_weight_tensor)
        w_sum = torch.clamp(weights.sum(), min=eps)
        loss_grpo = -(weights * l_clip).sum() / w_sum

        # KL penalty (paper: sum w_g [log p_old - log p]).
        loss_kl = (weights * (torch.log(p_ref) - torch.log(p))).sum() / w_sum
        try:
            self.last_batch_kl = float(loss_kl.detach().cpu().item())
        except Exception:
            self.last_batch_kl = None

        # Boundary preservation (paper: Huber vs theta_init for boundary entities).
        loss_boundary = torch.tensor(0.0, device=device)
        if getattr(self.args, "lambda_boundary", 0.0) and self.boundary_entities.numel() > 0:
            if self.init_ent_embeddings.numel() == 0:
                self.save_initial_embeddings()
            boundary_entities = self.boundary_entities
            max_b = int(getattr(self.args, "boundary_batch_size", 4096) or 0)
            if max_b > 0 and boundary_entities.numel() > max_b:
                # Random subset (with replacement) to bound compute cost.
                idx = torch.randint(0, boundary_entities.numel(), (max_b,), device=device)
                boundary_entities = boundary_entities.index_select(0, idx)
            cur = torch.index_select(self.kge_model.ent_embeddings.weight, 0, boundary_entities)
            init = torch.index_select(self.init_ent_embeddings.to(cur.device), 0, boundary_entities)
            # Adaptive Huber delta based on embedding norms (per timestep).
            with torch.no_grad():
                delta_scale = float(getattr(self.args, "boundary_delta_scale", 1.0))
                delta_scale = max(1e-4, delta_scale)
                norms = torch.norm(init, dim=1)
                if norms.numel() > 0:
                    delta = delta_scale * torch.median(norms)
                else:
                    delta = torch.tensor(1.0, device=cur.device)
            loss_boundary = torch.nn.functional.smooth_l1_loss(cur, init, reduction="mean", beta=float(delta))

        # Optional retain-only negative sampling loss (paper Eq. optional augmentation).
        loss_neg = torch.tensor(0.0, device=device)
        if float(getattr(self.args, "lambda_neg", 0.0)) > 0 and retain_triples.numel() > 0:
            loss_neg = self._retain_negative_sampling_loss(retain_triples)

        total = (
            float(getattr(self.args, "grpo_lambda", 1.0)) * loss_grpo
            + float(getattr(self.args, "lambda_kl", 0.0)) * loss_kl
            + float(getattr(self.args, "lambda_boundary", 0.0)) * loss_boundary
            + float(getattr(self.args, "lambda_neg", 0.0)) * loss_neg
        )
        return total

    def _retain_negative_sampling_loss(self, retain_triples: torch.Tensor) -> torch.Tensor:
        """Retain-only KGE loss with uniform corruption (keeps global calibration)."""
        device = retain_triples.device
        neg_ratio = int(getattr(self.args, "neg_ratio", 0) or 0)
        if neg_ratio <= 0:
            return torch.tensor(0.0, device=device)

        pos = retain_triples.long()
        bsz = pos.size(0)
        ent_num = int(getattr(self.kg, "ent_num", 0) or 0)
        if ent_num <= 0:
            return torch.tensor(0.0, device=device)

        # Generate corrupted negatives.
        neg = pos.repeat_interleave(neg_ratio, dim=0).clone()
        corrupt_heads = torch.randint(0, ent_num, (neg.size(0),), device=device)
        corrupt_tails = torch.randint(0, ent_num, (neg.size(0),), device=device)
        mask = torch.rand((neg.size(0),), device=device) > 0.5
        neg[mask, 0] = corrupt_heads[mask]
        neg[~mask, 2] = corrupt_tails[~mask]

        all_triples = torch.cat([pos, neg], dim=0)
        labels = torch.cat(
            [torch.ones((bsz,), device=device), -torch.ones((neg.size(0),), device=device)],
            dim=0,
        )
        return self.kge_model.loss(all_triples[:, 0], all_triples[:, 1], all_triples[:, 2], labels)

    def conflict_aware_projection_step(
        self,
        *,
        forget_triples: torch.Tensor,
        retain_triples: torch.Tensor,
        group_ids_forget: torch.Tensor = None,
        group_ids_retain: torch.Tensor = None,
        group_weight_tensor: torch.Tensor = None,
    ) -> None:
        """Periodic conflict-aware projection (paper: every K steps).

        Approximation: uses batch-local retain candidates in the same relation group and
        sharing an endpoint with each forget triple, prunes to top-N by current plausibility.
        """
        if forget_triples is None or retain_triples is None:
            return
        if forget_triples.numel() == 0 or retain_triples.numel() == 0:
            return

        device = self.args.device
        forget_triples = forget_triples.to(device)
        retain_triples = retain_triples.to(device)
        if group_ids_forget is not None:
            group_ids_forget = group_ids_forget.to(device)
        if group_ids_retain is not None:
            group_ids_retain = group_ids_retain.to(device)

        ent, rel = self.kge_model.embedding()
        raw_retain = self._triple_logits(retain_triples, ent_emb=ent, rel_emb=rel).detach()

        top_n = int(getattr(self.args, "projection_top_n", 10) or 10)
        proj_lambda = float(getattr(self.args, "projection_lambda", 0.5))
        eps = 1e-8

        # Optional: limit the number of forget triples to project (keeps projection cheap).
        max_forget = int(getattr(self.args, "projection_forget_batch_size", 128) or 0)
        if max_forget > 0 and forget_triples.size(0) > max_forget:
            raw_forget = self._triple_logits(forget_triples, ent_emb=ent, rel_emb=rel).detach()
            k = min(max_forget, raw_forget.numel())
            try:
                keep_idx = torch.topk(raw_forget, k=k, largest=True).indices
            except NotImplementedError:
                keep_idx = torch.topk(raw_forget.detach().cpu(), k=k, largest=True).indices.to(device)
            forget_triples = forget_triples.index_select(0, keep_idx)
            if group_ids_forget is not None:
                group_ids_forget = group_ids_forget.index_select(0, keep_idx)

        selected_forget = []
        selected_retain = []
        selected_forget_group_ids = []
        # Avoid iterating over device tensors directly (MPS sync); loop over a CPU copy of ids.
        forget_cpu = forget_triples.detach().cpu()
        group_ids_forget_cpu = group_ids_forget.detach().cpu() if group_ids_forget is not None else None
        for row_i in range(forget_cpu.size(0)):
            fh, fr, ftail = (int(forget_cpu[row_i, 0]), int(forget_cpu[row_i, 1]), int(forget_cpu[row_i, 2]))
            # Same semantic group (fallback to relation if no group ids provided).
            if group_ids_retain is not None and group_ids_forget_cpu is not None:
                f_gid = int(group_ids_forget_cpu[row_i])
                same_group = group_ids_retain == f_gid
                if not bool(same_group.any()):
                    continue
                rel_mask = same_group
            else:
                same_rel = retain_triples[:, 1] == fr
                if not bool(same_rel.any()):
                    continue
                rel_mask = same_rel
            # Share endpoint.
            share = (retain_triples[:, 0] == fh) | (retain_triples[:, 2] == fh) | (retain_triples[:, 0] == ftail) | (retain_triples[:, 2] == ftail)
            mask = rel_mask & share
            if not bool(mask.any()):
                continue
            cand_idx = torch.where(mask)[0]
            cand_scores = raw_retain.index_select(0, cand_idx)
            # Higher raw logits = more plausible.
            topk = min(top_n, cand_scores.numel())
            try:
                best = cand_idx[torch.topk(cand_scores, k=topk, largest=True).indices]
            except NotImplementedError:
                # MPS-safe fallback: selection on CPU (no gradients needed).
                cand_idx_cpu = cand_idx.detach().cpu()
                cand_scores_cpu = cand_scores.detach().cpu()
                best_cpu = cand_idx_cpu[torch.topk(cand_scores_cpu, k=topk, largest=True).indices]
                best = best_cpu.to(device)
            selected_forget.append(forget_triples[row_i : row_i + 1])
            if group_ids_forget_cpu is not None:
                selected_forget_group_ids.append(int(group_ids_forget_cpu[row_i]))
            selected_retain.append(retain_triples.index_select(0, best))

        if not selected_forget or not selected_retain:
            return

        forget_sel = torch.cat(selected_forget, dim=0)
        retain_sel = torch.cat(selected_retain, dim=0)
        forget_sel_group_ids = None
        if selected_forget_group_ids:
            forget_sel_group_ids = torch.tensor(selected_forget_group_ids, dtype=torch.long, device=device)

        mode = str(getattr(self.args, "projection_update_mode", "subembed")).lower()

        # Dense projection (legacy): computes gradients on the full embedding matrices (very slow on MPS).
        # Keep as an opt-in fallback for debugging.
        if mode in {"dense", "dense_optimizer"}:
            if self.optimizer is None:
                return
            raw_f = self._triple_logits(forget_sel, ent_emb=ent, rel_emb=rel)
            raw_r = self._triple_logits(retain_sel, ent_emb=ent, rel_emb=rel)
            z_f = (raw_f - self.score_ema_mu) / (self.score_ema_sigma + 1e-8)
            z_r = (raw_r - self.score_ema_mu) / (self.score_ema_sigma + 1e-8)
            obj_f = (-z_f).mean()
            obj_r = (z_r).mean()
            params = [self.kge_model.ent_embeddings.weight, self.kge_model.rel_embeddings.weight]
            g_f = torch.autograd.grad(obj_f, params, retain_graph=False, allow_unused=True)
            g_r = torch.autograd.grad(obj_r, params, retain_graph=False, allow_unused=True)
            dot = torch.tensor(0.0, device=device)
            r_norm2 = torch.tensor(0.0, device=device)
            for gf, gr in zip(g_f, g_r):
                if gf is None or gr is None:
                    continue
                dot = dot + torch.sum(gf * gr)
                r_norm2 = r_norm2 + torch.sum(gr * gr)
            if r_norm2 <= eps or dot <= 0:
                return
            if forget_sel_group_ids is not None and group_weight_tensor is not None:
                w = self._group_weights_for_ids(forget_sel_group_ids, group_weight_tensor).mean().detach()
            else:
                w = torch.tensor(1.0, device=device)
            scale = proj_lambda * w * (dot / (r_norm2 + eps))
            self.optimizer.zero_grad(set_to_none=True)
            for param, gf, gr in zip(params, g_f, g_r):
                if gf is None or gr is None:
                    continue
                param.grad = gf - scale * gr
            self.optimizer.step()
            return

        # MPS-friendly projection: compute gradients only for the embeddings actually touched by the selected triples,
        # then apply a small manual update to those rows (avoids dense grads + huge optimizer passes).
        with torch.no_grad():
            lr = None
            if self.optimizer is not None and getattr(self.optimizer, "param_groups", None):
                try:
                    lr = float(self.optimizer.param_groups[0].get("lr", None))
                except Exception:
                    lr = None
            if lr is None:
                lr = float(getattr(self.args, "lr", 1e-3))

        # Build compact sets of ids on CPU (mapping is small; avoids repeated MPS sync).
        ent_ids_cpu = torch.unique(
            torch.cat(
                [
                    forget_sel[:, [0, 2]].reshape(-1),
                    retain_sel[:, [0, 2]].reshape(-1),
                ],
                dim=0,
            )
            .detach()
            .cpu()
        )
        rel_ids_cpu = torch.unique(torch.cat([forget_sel[:, 1], retain_sel[:, 1]], dim=0).detach().cpu())
        ent_ids_cpu, _ = torch.sort(ent_ids_cpu)
        rel_ids_cpu, _ = torch.sort(rel_ids_cpu)
        ent_ids = ent_ids_cpu.to(device)
        rel_ids = rel_ids_cpu.to(device)

        # Extract sub-embeddings and treat them as independent variables for projection gradients.
        ent_sub = self.kge_model.ent_embeddings.weight.index_select(0, ent_ids).detach().clone().requires_grad_(True)
        rel_sub = self.kge_model.rel_embeddings.weight.index_select(0, rel_ids).detach().clone().requires_grad_(True)

        def _map_ids(ids_cpu: torch.Tensor, universe_cpu: torch.Tensor) -> torch.Tensor:
            # universe_cpu is sorted unique; searchsorted gives position in universe.
            return torch.searchsorted(universe_cpu, ids_cpu.contiguous()).long()

        def _logits_with_subemb(triples: torch.Tensor) -> torch.Tensor:
            tri_cpu = triples.detach().cpu()
            h_sub = _map_ids(tri_cpu[:, 0], ent_ids_cpu).to(device)
            r_sub = _map_ids(tri_cpu[:, 1], rel_ids_cpu).to(device)
            t_sub = _map_ids(tri_cpu[:, 2], ent_ids_cpu).to(device)
            h = torch.index_select(ent_sub, 0, h_sub)
            r = torch.index_select(rel_sub, 0, r_sub)
            t = torch.index_select(ent_sub, 0, t_sub)
            scores = self.kge_model.score_fun(h, r, t)
            return self._scores_to_logits(scores)

        raw_f = _logits_with_subemb(forget_sel)
        raw_r = _logits_with_subemb(retain_sel)
        mu = self.score_ema_mu.to(device)
        sigma = self.score_ema_sigma.to(device)
        z_f = (raw_f - mu) / (sigma + 1e-8)
        z_r = (raw_r - mu) / (sigma + 1e-8)
        obj_f = (-z_f).mean()
        obj_r = (z_r).mean()

        g_f_ent, g_f_rel = torch.autograd.grad(obj_f, [ent_sub, rel_sub], retain_graph=False, allow_unused=True)
        g_r_ent, g_r_rel = torch.autograd.grad(obj_r, [ent_sub, rel_sub], retain_graph=False, allow_unused=True)
        if g_f_ent is None or g_f_rel is None or g_r_ent is None or g_r_rel is None:
            return

        dot = torch.sum(g_f_ent * g_r_ent) + torch.sum(g_f_rel * g_r_rel)
        r_norm2 = torch.sum(g_r_ent * g_r_ent) + torch.sum(g_r_rel * g_r_rel)
        if r_norm2 <= eps or dot <= 0:
            return

        if forget_sel_group_ids is not None and group_weight_tensor is not None:
            w = self._group_weights_for_ids(forget_sel_group_ids, group_weight_tensor).mean().detach()
        else:
            w = torch.tensor(1.0, device=device)
        scale = proj_lambda * w * (dot / (r_norm2 + eps))
        proj_ent = g_f_ent - scale * g_r_ent
        proj_rel = g_f_rel - scale * g_r_rel

        # Apply a manual SGD-style update only to the touched rows.
        with torch.no_grad():
            self.kge_model.ent_embeddings.weight[ent_ids] = self.kge_model.ent_embeddings.weight.index_select(0, ent_ids) - lr * proj_ent
            self.kge_model.rel_embeddings.weight[rel_ids] = self.kge_model.rel_embeddings.weight.index_select(0, rel_ids) - lr * proj_rel

    def preservation_loss(self):
        """Calculate knowledge preservation loss between current and old embeddings.

        Returns:
            Tensor: preservation loss value
        """
        # Get current and old embeddings
        ent_embeddings, rel_embeddings = self.embedding()
        old_data_ent_embeddings, old_data_rel_embeddings = self.old_embeddings()

        # Apply entity distillation mask if available
        if self.entity_distill_mask is not None:
            self.entity_distill_mask = self.entity_distill_mask.to(self.args.device)
            ent_embeddings = ent_embeddings * self.entity_distill_mask.unsqueeze(1)
            old_data_ent_embeddings = old_data_ent_embeddings * self.entity_distill_mask.unsqueeze(1)

        # Calculate Huber loss between current and old embeddings
        ent_distill_loss = self.huber_loss(ent_embeddings, old_data_ent_embeddings)
        rel_distill_loss = self.huber_loss(rel_embeddings, old_data_rel_embeddings)

        # Total distillation loss
        distill_loss = ent_distill_loss + rel_distill_loss

        return distill_loss

    def form_triple_groups(self, triples, weights=None):
        """Form groups of triples based on specified strategy.

        Args:
            triples: Tensor of triples to group
            weights: Optional tensor of weights for each triple

        Returns:
            List of (group_triples, group_weights) tuples
        """
        device = triples.device

        # Ensure weights are on same device
        if weights is not None:
            weights = weights.to(device)

        # Choose grouping strategy
        strategy_map = {
            'relation': self._relation_based_grouping,
            'entity': self._entity_neighborhood_grouping,
            'schema': self._schema_coherent_grouping,
            'batch': self._batch_grouping
        }

        grouping_func = strategy_map.get(self.args.grouping_strategy, self._batch_grouping)
        return grouping_func(triples, weights)

    def _relation_based_grouping(self, triples, weights=None):
        """Group triples that share the same relation type.

        Args:
            triples: Tensor of triples
            weights: Optional tensor of weights

        Returns:
            List of (group_triples, group_weights) tuples
        """
        groups = []
        relations = triples[:, 1]
        device = triples.device

        # Ensure weights are on same device
        if weights is not None:
            weights = weights.to(device)

        # Find unique relations
        unique_relations = torch.unique(relations)

        for rel in unique_relations:
            # Use boolean indexing to find all triples with this relation
            mask = (relations == rel)
            group_triples = triples[mask]

            # Skip small groups
            if len(group_triples) < 2:
                continue

            # Apply weights if provided
            group_weights = weights[mask] if weights is not None else None
            groups.append((group_triples, group_weights))

        return groups

    def _entity_neighborhood_grouping(self, triples, weights=None):
        """Group triples connected to the same entity.

        Args:
            triples: Tensor of triples
            weights: Optional tensor of weights

        Returns:
            List of (group_triples, group_weights) tuples
        """
        groups = []
        device = triples.device

        # Ensure weights are on same device
        if weights is not None:
            weights = weights.to(device)

        # Extract head and tail entities
        heads = triples[:, 0]
        tails = triples[:, 2]

        # Find unique entities
        all_entities = torch.cat([heads, tails])
        unique_entities = torch.unique(all_entities)

        for entity in unique_entities:
            # Find all triples where this entity appears as head or tail
            head_mask = (heads == entity)
            tail_mask = (tails == entity)
            mask = head_mask | tail_mask

            group_triples = triples[mask]

            # Skip small groups
            if len(group_triples) < 2:
                continue

            # Apply weights if provided
            group_weights = weights[mask] if weights is not None else None
            groups.append((group_triples, group_weights))

        return groups

    def _schema_coherent_grouping(self, triples, weights=None):
        """Group triples based on schema weight similarity.

        Args:
            triples: Tensor of triples
            weights: Optional tensor of weights

        Returns:
            List of (group_triples, group_weights) tuples
        """
        groups = []
        device = triples.device

        # Fall back to relation grouping if no weights
        if weights is None:
            return self._relation_based_grouping(triples, weights)

        # Ensure weights are on same device
        weights = weights.to(device)

        # Define weight ranges for grouping
        w_min, w_max = weights.min().item(), weights.max().item()
        num_buckets = 4
        bucket_size = (w_max - w_min) / max(1, num_buckets)

        for i in range(num_buckets):
            bucket_min = w_min + i * bucket_size
            bucket_max = bucket_min + bucket_size

            # Find all triples with weights in this range
            mask = (weights >= bucket_min) & (weights < bucket_max)
            group_triples = triples[mask]

            # Skip small groups
            if len(group_triples) < 2:
                continue

            group_weights = weights[mask]
            groups.append((group_triples, group_weights))

        return groups

    def _batch_grouping(self, triples, weights=None):
        """Simple batching into groups of fixed size.

        Args:
            triples: Tensor of triples
            weights: Optional tensor of weights

        Returns:
            List of (group_triples, group_weights) tuples
        """
        groups = []
        device = triples.device

        # Calculate group size with safety checks
        group_size = min(getattr(self.args, 'group_size_grpo', 128), triples.size(0))
        num_groups = max(1, triples.size(0) // max(1, group_size))

        for g in range(num_groups):
            start_idx = g * group_size
            end_idx = min(start_idx + group_size, triples.size(0))

            group_triples = triples[start_idx:end_idx]

            if weights is not None:
                group_weights = weights[start_idx:end_idx]
            else:
                group_weights = None

            groups.append((group_triples, group_weights))

        return groups

    def compute_gradient_projection(self, forget_grad, retain_grad, schema_weight, beta=0.5):
        """Implement the gradient projection to protect retained knowledge.

        Args:
            forget_grad: Gradient to forget a specific triple
            retain_grad: Gradient to retain related triples
            schema_weight: Schema importance weight
            beta: Projection strength parameter

        Returns:
            Projected gradient
        """
        # Calculate projection of forget_grad onto retain_grad
        dot_product = torch.sum(forget_grad * retain_grad)
        retain_norm_squared = torch.sum(retain_grad * retain_grad)

        # Avoid division by zero
        if retain_norm_squared < 1e-8:
            return forget_grad

        projection_magnitude = dot_product / retain_norm_squared
        projected_component = projection_magnitude * retain_grad

        # Apply schema-weighted projection
        projected_grad = forget_grad - beta * schema_weight * projected_component

        return projected_grad

    def compute_policy_ratio(self, current_scores, old_scores):
        """Compute ratio between current and old policies.

        Args:
            current_scores: Current scores tensor
            old_scores: Old scores tensor

        Returns:
            Policy ratio tensor
        """
        # In TransE, lower scores are better, so we negate before sigmoid
        current_prob = torch.sigmoid(-current_scores)
        old_prob = torch.sigmoid(-old_scores)

        # Calculate policy ratio (with epsilon to prevent division by zero)
        ratio = current_prob / (old_prob + 1e-8)

        return ratio

    def compute_advantage(self, rewards):
        """Compute advantage using the GRPO formula.

        Args:
            rewards: Rewards tensor

        Returns:
            Advantages tensor
        """
        # Normalize rewards within group
        mean_reward = torch.mean(rewards)
        std_reward = torch.std(rewards) + 1e-8
        advantages = (rewards - mean_reward) / std_reward

        return advantages

    def compute_kl_divergence(self, current_prob, old_prob):
        """Compute KL divergence between current and old policies.

        Args:
            current_prob: Current probability tensor
            old_prob: Old probability tensor

        Returns:
            KL divergence value
        """
        # KL(current || old)
        kl_div = (current_prob * torch.log((current_prob + 1e-8) / (old_prob + 1e-8))).mean()
        return kl_div

    @torch.no_grad()
    def schema_grpo_loss(self, pos_triples, pos_weights, neg_triples):
        """Optimized schema-aware GRPO loss calculation.

        Args:
            pos_triples: Positive triples tensor
            pos_weights: Schema weights tensor
            neg_triples: Negative triples tensor

        Returns:
            Tensor: GRPO loss value
        """
        # Check for valid inputs
        if pos_triples is None or neg_triples is None or pos_triples.size(0) == 0:
            print("ERROR! no pos or neg triples GRPO loss")
            exit(1)
            return torch.tensor(0.0, device=self.args.device)



        # Get embeddings
        ent_embeddings, rel_embeddings = self.kge_model.embedding()
        old_ent_embeddings, old_rel_embeddings = self.old_embeddings()

        # Add noise if embeddings are too similar
        if torch.norm(ent_embeddings[:10] - old_ent_embeddings[:10]) < 1e-3:
            old_ent_embeddings = old_ent_embeddings + 0.01 * torch.rand_like(old_ent_embeddings)
            old_rel_embeddings = old_rel_embeddings + 0.01 * torch.rand_like(old_rel_embeddings)

        # Group triples according to selected strategy
        matched_groups = self._create_matched_groups(
            pos_triples, neg_triples, pos_weights
        )

        # Calculate losses for each group
        losses = []
        for group_pos, group_neg, group_weights in matched_groups:
            # Skip empty groups
            if len(group_pos) == 0 or len(group_neg) == 0:
                continue

            # Get embeddings for current group
            pos_h, pos_r, pos_t = self._get_embeddings_batch(
                ent_embeddings, rel_embeddings, group_pos
            )
            old_pos_h, old_pos_r, old_pos_t = self._get_embeddings_batch(
                old_ent_embeddings, old_rel_embeddings, group_pos
            )
            neg_h, neg_r, neg_t = self._get_embeddings_batch(
                ent_embeddings, rel_embeddings, group_neg
            )
            old_neg_h, old_neg_r, old_neg_t = self._get_embeddings_batch(
                old_ent_embeddings, old_rel_embeddings, group_neg
            )

            # Compute scores
            pos_scores = self.kge_model.score_fun(pos_h, pos_r, pos_t)
            old_pos_scores = self.kge_model.score_fun(old_pos_h, old_pos_r, old_pos_t)
            neg_scores = self.kge_model.score_fun(neg_h, neg_r, neg_t)
            old_neg_scores = self.kge_model.score_fun(old_neg_h, old_neg_r, old_neg_t)

            # Add noise if scores are too similar
            if torch.norm(pos_scores - old_pos_scores) < 1e-4:
                pos_scores = pos_scores + 0.01 * torch.rand_like(pos_scores)
            if torch.norm(neg_scores - old_neg_scores) < 1e-4:
                neg_scores = neg_scores + 0.01 * torch.rand_like(neg_scores)

            # Compute policy probabilities
            pos_prob = torch.sigmoid(-pos_scores)
            old_pos_prob = torch.sigmoid(-old_pos_scores)
            neg_prob = torch.sigmoid(-neg_scores)
            old_neg_prob = torch.sigmoid(-old_neg_scores)

            # Compute rewards with schema weighting
            rewards = (pos_prob - neg_prob)
            if group_weights is not None:
                rewards = rewards * group_weights

            # Ensure rewards have variation
            if rewards.std() < 1e-4:
                rewards = rewards + 0.01 * torch.rand_like(rewards)

            # Compute policy ratios
            pos_ratios = self.compute_policy_ratio(pos_scores, old_pos_scores)
            neg_ratios = self.compute_policy_ratio(neg_scores, old_neg_scores)

            # Normalize advantages
            advantages = self.compute_advantage(rewards)

            # Compute clipped surrogate objectives
            epsilon = self.args.epsilon_grpo
            clipped_pos_ratios = torch.clamp(pos_ratios, 1.0 - epsilon, 1.0 + epsilon)
            clipped_neg_ratios = torch.clamp(neg_ratios, 1.0 - epsilon, 1.0 + epsilon)

            pos_surrogate = torch.min(pos_ratios * advantages, clipped_pos_ratios * advantages)
            neg_surrogate = torch.min(neg_ratios * -advantages, clipped_neg_ratios * -advantages)

            # Compute KL divergence
            kl_pos = self.compute_kl_divergence(pos_prob, old_pos_prob)
            kl_neg = self.compute_kl_divergence(neg_prob, old_neg_prob)

            # Compute loss
            group_loss = -pos_surrogate.mean() - neg_surrogate.mean() + self.args.beta_grpo * (kl_pos + kl_neg)
            losses.append(group_loss)

        # Compute average loss
        if losses:
            return torch.stack(losses).mean()
        return torch.tensor(0.0, device=pos_triples.device)

    def set_boundary_preservation_entities(self, pos_triples):
        # Extract all unique relation IDs from pos_triples
        relations = pos_triples[:, 1].unique()

        # For each relation, identify the entities connected by this relation
        boundary_entities = set()

        for relation in relations:
            # Find all triples with this relation
            rel_mask = (pos_triples[:, 1] == relation)
            rel_triples = pos_triples[rel_mask]

            # Extract head and tail entities
            heads = rel_triples[:, 0].tolist()
            tails = rel_triples[:, 2].tolist()

            # Add these entities to boundary entities
            boundary_entities.update(heads + tails)

        # Convert to tensor
        return torch.tensor(list(boundary_entities),
                            dtype=torch.long, device=self.args.device)

    def boundary_preservation_loss(self, preservation_entities):
        """Calculate boundary preservation loss to enhance recall of knowledge in the forgetting boundary.

        Args:
            preservation_entities: Optional tensor of entity IDs in the distillation boundary.
                             If None, uses self.boundary_distill_entities.

        Returns:
            Tensor: Boundary distillation loss value
        """
        # Use stored boundary entities if none provided

        # Get current and reference embeddings
        current_embeddings = self.kge_model.ent_embeddings.weight

        # Check if reference model embeddings are available
        if not hasattr(self, 'reference_model') or self.reference_model is None:
            # Fall back to using old embeddings as reference
            reference_embeddings, _ = self.old_embeddings()
        else:
            # Use dedicated reference model
            reference_embeddings = self.kge_model.ent_embeddings.weight

        # Extract embeddings for distillation entities
        preservation_entities = preservation_entities.to(self.args.device)
        e = torch.index_select(current_embeddings, 0, preservation_entities)
        e_ref = torch.index_select(reference_embeddings, 0, preservation_entities)

        # Calculate element-wise differences
        diffs = e - e_ref
        diff_norms = torch.norm(diffs, dim=1)

        # Apply Huber-like loss as per Equation 15
        mask_small = (diff_norms <= 1.0)
        mask_large = (diff_norms > 1.0)

        # Compute loss for each case
        loss_small = 0.5 * torch.sum(diffs[mask_small] ** 2, dim=1)
        loss_large = diff_norms[mask_large] - 0.5

        # Combine losses
        total_loss = torch.zeros_like(diff_norms)
        total_loss[mask_small] = loss_small
        total_loss[mask_large] = loss_large

        # Average loss over all boundary entities
        return torch.mean(total_loss)

    def _create_matched_groups(self, pos_triples, neg_triples, pos_weights):
        """Create groups of positive and negative triples.

        Args:
            pos_triples: Positive triples tensor
            neg_triples: Negative triples tensor
            pos_weights: Schema weights tensor

        Returns:
            List of (group_pos, group_neg, group_weights) tuples
        """
        if self.args.grouping_strategy != 'batch':
            # Group by selected strategy
            triple_groups = self.form_triple_groups(pos_triples, pos_weights)

            matched_groups = []
            for group_pos, group_weights in triple_groups:
                # Find matching negative triples
                group_indices = torch.zeros(pos_triples.size(0), dtype=torch.bool, device=pos_triples.device)

                for pos_triple in group_pos:
                    # Find index of this triple in the original pos_triples
                    matches = ((pos_triples == pos_triple).all(dim=1))
                    group_indices = group_indices | matches

                group_neg = neg_triples[group_indices]
                matched_groups.append((group_pos, group_neg, group_weights))
        else:
            # Process in larger groups (simple batching)
            group_size = min(self.args.group_size_grpo, pos_triples.size(0))
            num_groups = max(1, pos_triples.size(0) // group_size)

            matched_groups = []
            for g in range(num_groups):
                start_idx = g * group_size
                end_idx = min(start_idx + group_size, pos_triples.size(0))

                group_pos = pos_triples[start_idx:end_idx]
                group_neg = neg_triples[start_idx:end_idx]
                group_weights = pos_weights[start_idx:end_idx] if pos_weights is not None else None

                matched_groups.append((group_pos, group_neg, group_weights))

        return matched_groups

    def _get_embeddings_batch(self, ent_emb, rel_emb, triples):
        """Efficient batched embedding lookup.

        Args:
            ent_emb: Entity embeddings
            rel_emb: Relation embeddings
            triples: Triples tensor

        Returns:
            Tuple of (h, r, t) embeddings
        """
        h = torch.index_select(ent_emb, 0, triples[:, 0])
        r = torch.index_select(rel_emb, 0, triples[:, 1])
        t = torch.index_select(ent_emb, 0, triples[:, 2])
        return h, r, t

    def combined_loss(self, head, relation, tail, label, pos_triples, neg_triples, pos_weights=None):
        """Combined loss function with schema guidance.

        Args:
            head: Head entity embeddings
            relation: Relation embeddings
            tail: Tail entity embeddings
            label: Triple labels
            pos_triples: Positive triples tensor
            neg_triples: Negative triples tensor
            pos_weights: Schema weights tensor

        Returns:
            Tensor: Combined loss value
        """
        # Base model loss
        base_loss = self.kge_model.loss(head, relation, tail, label)

        # Schema-guided GRPO loss
        grpo_loss = self.schema_grpo_loss(pos_triples, pos_weights, neg_triples)

        # Apply lambda weighting
        weighted_grpo_loss = self.args.grpo_lambda * grpo_loss

        preservation_loss = -1  # it means not used

        # Add distillation loss
        if self.args.use_distill_loss:
            preservation_loss = self.boundary_preservation_loss(
                self.set_boundary_preservation_entities(pos_triples))  # self.preservation_loss()
            weighted_distill_loss = self.args.distill_lambda * preservation_loss
            total_loss = base_loss + weighted_grpo_loss + weighted_distill_loss
        else:
            # Combined loss
            total_loss = base_loss + weighted_grpo_loss  # + weighted_distill_loss

        # Periodic logging
        # self._log_loss_values(base_loss, grpo_loss, preservation_loss, total_loss)

        return total_loss

    def _log_loss_values(self, base_loss, grpo_loss, distill_loss, total_loss):
        """Log loss values periodically.

        Args:
            base_loss: Base loss value
            grpo_loss: GRPO loss value
            distill_loss: Distillation loss value
            total_loss: Total combined loss value
        """
        self._loss_counter += 1
        if self._loss_counter % 10 == 0:
            print(
                f" Loss: Base={base_loss.item():.4f}, GRPO={grpo_loss.item():.4f}, " +
                f" Distill={distill_loss.item():.4f}, Total={total_loss.item():.4f}")

    def gradient_guided_optimization_step(self, forget_triples, retain_triples, schema_weights=None):
        """Implement explicit gradient projection for selective forgetting.

        Args:
            forget_triples: Triples to forget
            retain_triples: Triples to retain
            schema_weights: Schema weights tensor

        Returns:
            dict: Result with forget_loss
        """
        # Make sure we're in training mode
        self.train()

        # Enable gradient computation for all parameters
        for param in self.parameters():
            param.requires_grad = True

        # Convert triples to tensors if they aren't already
        if not torch.is_tensor(forget_triples):
            forget_triples = torch.tensor(forget_triples, device=self.args.device)
        if not torch.is_tensor(retain_triples):
            retain_triples = torch.tensor(retain_triples, device=self.args.device)

        # 1. Compute forget gradients
        try:
            # Ensure we're working with tensors that require gradients
            forget_h, forget_r, forget_t = self._get_embeddings_batch(
                self.kge_model.ent_embeddings.weight, self.kge_model.rel_embeddings.weight, forget_triples
            )

            # Make sure tensors require gradients
            if not forget_h.requires_grad:
                forget_h = forget_h.detach().requires_grad_(True)
            if not forget_r.requires_grad:
                forget_r = forget_r.detach().requires_grad_(True)
            if not forget_t.requires_grad:
                forget_t = forget_t.detach().requires_grad_(True)

            forget_scores = self.kge_model.score_fun(forget_h, forget_r, forget_t)
            forget_loss = torch.mean(forget_scores)

            self.zero_grad()
            forget_loss.backward(retain_graph=True)

            # Store forget gradients
            forget_grads = {name: param.grad.clone() if param.grad is not None else torch.zeros_like(param)
                            for name, param in self.named_parameters()}

            # 2. Compute retain gradients
            self.zero_grad()
            retain_h, retain_r, retain_t = self._get_embeddings_batch(
                self.kge_model.ent_embeddings.weight, self.kge_model.rel_embeddings.weight, retain_triples
            )

            # Make sure tensors require gradients
            if not retain_h.requires_grad:
                retain_h = retain_h.detach().requires_grad_(True)
            if not retain_r.requires_grad:
                retain_r = retain_r.detach().requires_grad_(True)
            if not retain_t.requires_grad:
                retain_t = retain_t.detach().requires_grad_(True)

            retain_scores = self.kge_model.score_fun(retain_h, retain_r, retain_t)
            retain_loss = torch.mean(retain_scores)

            self.zero_grad()
            retain_loss.backward()

            # Store retain gradients
            retain_grads = {name: param.grad.clone() if param.grad is not None else torch.zeros_like(param)
                            for name, param in self.named_parameters()}

            # 3. Apply schema-weighted gradient projection
            beta = self.args.gradient_projection_weight  # Projection strength
            projected_grads = {}

            # Handle schema weights properly
            if schema_weights is not None:
                if isinstance(schema_weights, float):
                    avg_schema_weight = torch.tensor(schema_weights, device=self.args.device)
                elif torch.is_tensor(schema_weights):
                    avg_schema_weight = torch.mean(schema_weights)
                else:
                    # Default if schema_weights is neither tensor nor float
                    avg_schema_weight = torch.tensor(1.0, device=self.args.device)
            else:
                avg_schema_weight = torch.tensor(1.0, device=self.args.device)

            for name in forget_grads:
                if name in retain_grads:
                    # Apply gradient projection
                    projected_grads[name] = self.compute_gradient_projection(
                        forget_grads[name],
                        retain_grads[name],
                        avg_schema_weight,
                        beta
                    )

            # 4. Apply projected gradients
            self.zero_grad()
            for name, param in self.named_parameters():
                if name in projected_grads:
                    if param.grad is None:
                        param.grad = projected_grads[name].to(param.device)
                    else:
                        param.grad.copy_(projected_grads[name])

            # 5. Update parameters
            self.optimizer.step()

            return {'forget_loss': forget_loss.item(), 'projection_applied': True}

        except Exception as e:
            print(f"ERROR in gradient-guided optimization: {e}")
            # Return a default result with error information
            return {'forget_loss': 0.0, 'error': str(e), 'projection_applied': False}
