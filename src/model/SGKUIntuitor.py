import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utilities.utilities import *


class SGKUIntuitor(nn.Module):
    """Enhanced Schema-guided Knowledge Unlearning with INTUITOR's self-certainty approach."""

    def __init__(self, args, kg, kge_model_class, schema_store=None):
        """
        Initialize enhanced SGKU model with INTUITOR capabilities.

        Args:
            args: Arguments with model configuration
            kg: Knowledge graph instance
            kge_model_class: Base KGE model class
            schema_store: Optional schema store for pattern retrieval
        """
        super(SGKUIntuitor, self).__init__()

        # Initialize the base KGE model
        self.kge_model_class = kge_model_class
        self.kge_model = self.kge_model_class(args, kg)

        self.args = args
        self.kg = kg
        self.schema_store = schema_store
        self.huber_loss = nn.HuberLoss(reduction="sum")
        self.entity_distill_mask = None

        # Tracking variables
        self._last_warning_time = 0
        self._loss_counter = 0

        # Configure hyperparameters
        self._set_hyperparameters()

        # INTUITOR-specific parameters
        self.vocab_size = getattr(kg, 'n_entity', getattr(kg, 'ent_num', len(getattr(kg, 'entity2id', {}))))

    def _set_hyperparameters(self):
        """Set GRPO and INTUITOR hyperparameters with defaults."""
        # Main GRPO hyperparameters
        self.args.epsilon_grpo = getattr(self.args, 'epsilon_grpo', 0.2)
        self.args.beta_grpo = getattr(self.args, 'beta_grpo', 0.001)
        self.args.group_size_grpo = getattr(self.args, 'group_size_grpo', 128)
        self.args.grpo_lambda = getattr(self.args, 'grpo_lambda', 0.5)
        self.args.preservation_lambda = getattr(self.args, 'preservation_lambda', 0.5)

        # INTUITOR-specific parameters
        self.args.use_self_certainty = getattr(self.args, 'use_self_certainty', True)
        self.args.certainty_lambda = getattr(self.args, 'certainty_lambda', 1.0)
        self.args.certainty_threshold = getattr(self.args, 'certainty_threshold', 0.1)

        # Grouping strategy
        self.args.grouping_strategy = getattr(self.args, 'grouping_strategy', 'schema')

    def compute_self_certainty(self, scores, mask=None):
        """
        Compute self-certainty as KL divergence from uniform distribution.
        Following INTUITOR's Equation 4.

        Args:
            scores: Model scores tensor [batch_size]
            mask: Optional mask for valid positions

        Returns:
            Self-certainty value (higher = more confident)
        """
        # Convert scores to probability distribution using softmax
        # For TransE-style models, we need to convert distances to probabilities
        logits = -scores  # Convert distance to preference (lower distance = higher preference)
        probs = F.softmax(logits.unsqueeze(-1), dim=-1)

        # Create uniform distribution
        uniform_probs = torch.ones_like(probs) / probs.size(-1)

        # Compute KL divergence: KL(Uniform || Model)
        # Self-certainty = -1/|o| * sum(log(|V| * p(token)))
        log_probs = torch.log(probs + 1e-8)
        certainty = -torch.mean(log_probs)

        return certainty

    def compute_schema_weighted_certainty(self, triples, scores, schema_weights=None):
        """
        Compute schema-weighted self-certainty for a batch of triples.

        Args:
            triples: Batch of triples [batch_size, 3]
            scores: Model scores for triples [batch_size]
            schema_weights: Optional schema weights [batch_size]

        Returns:
            Weighted certainty scores [batch_size]
        """
        certainty_scores = []

        for i, triple in enumerate(triples):
            # Compute base self-certainty
            base_certainty = self.compute_self_certainty(scores[i:i + 1])

            # Apply schema weighting if available
            if schema_weights is not None:
                weighted_certainty = base_certainty * schema_weights[i]
            else:
                weighted_certainty = base_certainty

            certainty_scores.append(weighted_certainty)

        return torch.stack(certainty_scores)

    def compute_intuitor_rewards(self, pos_triples, neg_triples, pos_weights=None):
        """
        Compute rewards using INTUITOR's self-certainty approach.

        Args:
            pos_triples: Positive triples to retain [batch_size, 3]
            neg_triples: Negative triples to forget [batch_size, 3]
            pos_weights: Optional schema weights [batch_size]

        Returns:
            Rewards tensor for GRPO [batch_size]
        """
        # Get current embeddings
        ent_embeddings, rel_embeddings = self.kge_model.embedding()

        # Compute scores for positive triples (to retain)
        pos_h, pos_r, pos_t = self._get_embeddings_batch(ent_embeddings, rel_embeddings, pos_triples)
        pos_scores = self.kge_model.score_fun(pos_h, pos_r, pos_t)

        # Compute scores for negative triples (to forget)
        neg_h, neg_r, neg_t = self._get_embeddings_batch(ent_embeddings, rel_embeddings, neg_triples)
        neg_scores = self.kge_model.score_fun(neg_h, neg_r, neg_t)

        # Compute self-certainty for positive triples (encourage high confidence)
        pos_certainty = self.compute_schema_weighted_certainty(pos_triples, pos_scores, pos_weights)

        # Compute self-certainty for negative triples (discourage high confidence)
        neg_certainty = self.compute_schema_weighted_certainty(neg_triples, neg_scores, pos_weights)

        # INTUITOR-style reward: R_ai = f+_θ(ai) - f-_θ(ai)
        # For retention: positive certainty reward
        # For forgetting: negative certainty reward
        rewards = pos_certainty - neg_certainty

        return rewards

    def enhanced_schema_grpo_loss(self, pos_triples, pos_weights, neg_triples):
        """
        Enhanced GRPO loss using INTUITOR's self-certainty rewards.

        Args:
            pos_triples: Positive triples tensor
            pos_weights: Schema weights tensor
            neg_triples: Negative triples tensor

        Returns:
            Enhanced GRPO loss value
        """
        if pos_triples is None or neg_triples is None or pos_triples.size(0) == 0:
            print("ERROR! no pos or neg triples in enhanced GRPO loss")
            return torch.tensor(0.0, device=self.args.device)

        # Get current and old embeddings
        ent_embeddings, rel_embeddings = self.kge_model.embedding()
        old_ent_embeddings, old_rel_embeddings = self.old_embeddings()

        # Add noise if embeddings are too similar
        if torch.norm(ent_embeddings[:10] - old_ent_embeddings[:10]) < 1e-3:
            old_ent_embeddings = old_ent_embeddings + 0.01 * torch.rand_like(old_ent_embeddings)
            old_rel_embeddings = old_rel_embeddings + 0.01 * torch.rand_like(old_rel_embeddings)

        # Group triples according to selected strategy
        matched_groups = self._create_matched_groups(pos_triples, neg_triples, pos_weights)

        losses = []
        for group_pos, group_neg, group_weights in matched_groups:
            if len(group_pos) == 0 or len(group_neg) == 0:
                continue

            # Get embeddings for current group
            pos_h, pos_r, pos_t = self._get_embeddings_batch(ent_embeddings, rel_embeddings, group_pos)
            old_pos_h, old_pos_r, old_pos_t = self._get_embeddings_batch(old_ent_embeddings, old_rel_embeddings,
                                                                         group_pos)
            neg_h, neg_r, neg_t = self._get_embeddings_batch(ent_embeddings, rel_embeddings, group_neg)
            old_neg_h, old_neg_r, old_neg_t = self._get_embeddings_batch(old_ent_embeddings, old_rel_embeddings,
                                                                         group_neg)

            # Compute current and old scores
            pos_scores = self.kge_model.score_fun(pos_h, pos_r, pos_t)
            old_pos_scores = self.kge_model.score_fun(old_pos_h, old_pos_r, old_pos_t)
            neg_scores = self.kge_model.score_fun(neg_h, neg_r, neg_t)
            old_neg_scores = self.kge_model.score_fun(old_neg_h, old_neg_r, old_neg_t)

            # Add noise if scores are too similar
            if torch.norm(pos_scores - old_pos_scores) < 1e-4:
                pos_scores = pos_scores + 0.01 * torch.rand_like(pos_scores)
            if torch.norm(neg_scores - old_neg_scores) < 1e-4:
                neg_scores = neg_scores + 0.01 * torch.rand_like(neg_scores)

            # Compute INTUITOR-style rewards using self-certainty
            if self.args.use_self_certainty:
                rewards = self.compute_intuitor_rewards(group_pos, group_neg, group_weights)
            else:
                # Fallback to original probability-based rewards
                pos_prob = torch.sigmoid(-pos_scores)
                neg_prob = torch.sigmoid(-neg_scores)
                rewards = pos_prob - neg_prob
                if group_weights is not None:
                    rewards = rewards * group_weights

            # Ensure rewards have variation
            if rewards.std() < 1e-4:
                rewards = rewards + 0.01 * torch.rand_like(rewards)

            # Compute schema-weighted advantages (INTUITOR Equation 4)
            mean_reward = torch.mean(rewards)
            advantages = rewards - mean_reward

            # Apply schema weighting to advantages
            if group_weights is not None:
                advantages = group_weights * advantages

            # Compute policy ratios
            pos_ratios = self.compute_policy_ratio(pos_scores, old_pos_scores)
            neg_ratios = self.compute_policy_ratio(neg_scores, old_neg_scores)

            # Clipped surrogate objectives (INTUITOR Equation 5)
            epsilon = self.args.epsilon_grpo
            clipped_pos_ratios = torch.clamp(pos_ratios, 1.0 - epsilon, 1.0 + epsilon)
            clipped_neg_ratios = torch.clamp(neg_ratios, 1.0 - epsilon, 1.0 + epsilon)

            pos_surrogate = torch.min(pos_ratios * advantages, clipped_pos_ratios * advantages)
            neg_surrogate = torch.min(neg_ratios * -advantages, clipped_neg_ratios * -advantages)

            # KL regularization following INTUITOR approach
            pos_prob = torch.sigmoid(-pos_scores)
            old_pos_prob = torch.sigmoid(-old_pos_scores)
            neg_prob = torch.sigmoid(-neg_scores)
            old_neg_prob = torch.sigmoid(-old_neg_scores)

            kl_pos = self.compute_kl_divergence(pos_prob, old_pos_prob)
            kl_neg = self.compute_kl_divergence(neg_prob, old_neg_prob)

            # Final GRPO loss (INTUITOR Equation 6)
            group_loss = -pos_surrogate.mean() - neg_surrogate.mean() + self.args.beta_grpo * (kl_pos + kl_neg)
            losses.append(group_loss)

        if losses:
            return torch.stack(losses).mean()
        return torch.tensor(0.0, device=pos_triples.device)

    def emergent_reasoning_loss(self, triples, target_certainty=0.8):
        """
        Encourage emergent structured reasoning by rewarding high-certainty reasoning paths.
        Inspired by INTUITOR's emergent long-form reasoning.

        Args:
            triples: Input triples for reasoning
            target_certainty: Target certainty threshold

        Returns:
            Reasoning encouragement loss
        """
        ent_embeddings, rel_embeddings = self.kge_model.embedding()
        h, r, t = self._get_embeddings_batch(ent_embeddings, rel_embeddings, triples)
        scores = self.kge_model.score_fun(h, r, t)

        # Compute self-certainty for reasoning paths
        certainty = self.compute_schema_weighted_certainty(triples, scores)

        # Encourage reasoning paths that exceed target certainty
        reasoning_loss = F.relu(target_certainty - certainty).mean()

        return reasoning_loss

    def intuitor_combined_loss(self, head, relation, tail, label, pos_triples, neg_triples, pos_weights=None):
        """
        Enhanced combined loss with INTUITOR integration.

        Args:
            head: Head entity embeddings
            relation: Relation embeddings
            tail: Tail entity embeddings
            label: Triple labels
            pos_triples: Positive triples tensor
            neg_triples: Negative triples tensor
            pos_weights: Schema weights tensor

        Returns:
            Combined loss with INTUITOR enhancements
        """
        # Base KGE loss
        base_loss = self.kge_model.loss(head, relation, tail, label)

        # Enhanced GRPO loss with self-certainty rewards
        grpo_loss = self.enhanced_schema_grpo_loss(pos_triples, pos_weights, neg_triples)
        weighted_grpo_loss = self.args.grpo_lambda * grpo_loss

        # Emergent reasoning loss (inspired by INTUITOR's structured reasoning)
        reasoning_loss = self.emergent_reasoning_loss(pos_triples)
        weighted_reasoning_loss = self.args.certainty_lambda * reasoning_loss

        # Boundary preservation loss
        preservation_loss = torch.tensor(0.0, device=self.args.device)
        if self.args.use_distill_loss:
            preservation_entities = self.set_boundary_preservation_entities(pos_triples)
            preservation_loss = self.boundary_preservation_loss(preservation_entities)
            weighted_preservation_loss = self.args.preservation_lambda * preservation_loss

            total_loss = (base_loss + weighted_grpo_loss +
                          weighted_reasoning_loss + weighted_preservation_loss)
        else:
            total_loss = base_loss + weighted_grpo_loss + weighted_reasoning_loss

        return total_loss

    def generate_explanations(self, triples, top_k=3):
        """
        Generate explanations for unlearning decisions using self-certainty.
        Inspired by INTUITOR's emergent reasoning capabilities.

        Args:
            triples: Triples to explain
            top_k: Number of top explanations to return

        Returns:
            List of explanation dictionaries
        """
        explanations = []

        with torch.no_grad():
            ent_embeddings, rel_embeddings = self.kge_model.embedding()

            for triple in triples[:top_k]:
                h, r, t = self._get_embeddings_batch(
                    ent_embeddings, rel_embeddings, triple.unsqueeze(0)
                )
                score = self.kge_model.score_fun(h, r, t)
                certainty = self.compute_self_certainty(score)

                # Get schema pattern
                schema_pattern = self.get_schema_pattern(triple)

                explanation = {
                    'triple': triple.tolist(),
                    'certainty': certainty.item(),
                    'schema_pattern': schema_pattern,
                    'action': 'retain' if certainty > self.args.certainty_threshold else 'forget',
                    'confidence': 'high' if certainty > 0.5 else 'low'
                }
                explanations.append(explanation)

        return explanations

    def get_schema_pattern(self, triple):
        """Extract schema pattern from a triple."""
        if self.schema_store:
            return self.schema_store.get_pattern(triple)
        else:
            # Fallback: use relation as pattern
            return f"pattern_{triple[1].item()}"

    # Keep all the original SGKU methods that don't need modification
    def save_embeddings(self):
        """Save current embeddings as reference for GRPO and distillation."""
        for name, param in self.named_parameters():
            name = name.replace('.', '_')
            if param.requires_grad:
                value = param.data
                self.register_buffer(f'old_data_{name}', value.clone().detach())

    def embedding(self):
        """Get current embeddings."""
        return self.kge_model.ent_embeddings.weight, self.kge_model.rel_embeddings.weight

    def old_embeddings(self):
        """Get old embeddings for GRPO."""
        old_data_ent_embeddings_weight = None
        old_data_rel_embeddings_weight = None

        for name, value in self.named_buffers():
            if name == "old_data_ent_embeddings_weight":
                old_data_ent_embeddings_weight = value
            elif name == "old_data_rel_embeddings_weight":
                old_data_rel_embeddings_weight = value

        if old_data_ent_embeddings_weight is None:
            old_data_ent_embeddings_weight = self.kge_model.ent_embeddings.weight.clone().detach()
        if old_data_rel_embeddings_weight is None:
            old_data_rel_embeddings_weight = self.kge_model.rel_embeddings.weight.clone().detach()

        return old_data_ent_embeddings_weight, old_data_rel_embeddings_weight

    def form_triple_groups(self, triples, weights=None):
        """Form groups of triples based on specified strategy."""
        device = triples.device
        if weights is not None:
            weights = weights.to(device)

        strategy_map = {
            'relation': self._relation_based_grouping,
            'entity': self._entity_neighborhood_grouping,
            'schema': self._schema_coherent_grouping,
            'batch': self._batch_grouping
        }

        grouping_func = strategy_map.get(self.args.grouping_strategy, self._batch_grouping)
        return grouping_func(triples, weights)

    def _relation_based_grouping(self, triples, weights=None):
        """Group triples that share the same relation type."""
        groups = []
        relations = triples[:, 1]
        device = triples.device

        if weights is not None:
            weights = weights.to(device)

        unique_relations = torch.unique(relations)

        for rel in unique_relations:
            mask = (relations == rel)
            group_triples = triples[mask]
            if len(group_triples) < 2:
                continue
            group_weights = weights[mask] if weights is not None else None
            groups.append((group_triples, group_weights))

        return groups

    def _entity_neighborhood_grouping(self, triples, weights=None):
        """Group triples connected to the same entity."""
        groups = []
        device = triples.device

        if weights is not None:
            weights = weights.to(device)

        heads = triples[:, 0]
        tails = triples[:, 2]
        all_entities = torch.cat([heads, tails])
        unique_entities = torch.unique(all_entities)

        for entity in unique_entities:
            head_mask = (heads == entity)
            tail_mask = (tails == entity)
            mask = head_mask | tail_mask
            group_triples = triples[mask]

            if len(group_triples) < 2:
                continue

            group_weights = weights[mask] if weights is not None else None
            groups.append((group_triples, group_weights))

        return groups

    def _schema_coherent_grouping(self, triples, weights=None):
        """Group triples based on schema weight similarity."""
        groups = []
        device = triples.device

        if weights is None:
            return self._relation_based_grouping(triples, weights)

        weights = weights.to(device)
        w_min, w_max = weights.min().item(), weights.max().item()
        num_buckets = 4
        bucket_size = (w_max - w_min) / max(1, num_buckets)

        for i in range(num_buckets):
            bucket_min = w_min + i * bucket_size
            bucket_max = bucket_min + bucket_size

            mask = (weights >= bucket_min) & (weights < bucket_max)
            group_triples = triples[mask]

            if len(group_triples) < 2:
                continue

            group_weights = weights[mask]
            groups.append((group_triples, group_weights))

        return groups

    def _batch_grouping(self, triples, weights=None):
        """Simple batching into groups of fixed size."""
        groups = []
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

    def compute_policy_ratio(self, current_scores, old_scores):
        """Compute ratio between current and old policies."""
        current_prob = torch.sigmoid(-current_scores)
        old_prob = torch.sigmoid(-old_scores)
        ratio = current_prob / (old_prob + 1e-8)
        return ratio

    def compute_advantage(self, rewards):
        """Compute advantage using the GRPO formula."""
        mean_reward = torch.mean(rewards)
        std_reward = torch.std(rewards) + 1e-8
        advantages = (rewards - mean_reward) / std_reward
        return advantages

    def compute_kl_divergence(self, current_prob, old_prob):
        """Compute KL divergence between current and old policies."""
        kl_div = (current_prob * torch.log((current_prob + 1e-8) / (old_prob + 1e-8))).mean()
        return kl_div

    def set_boundary_preservation_entities(self, pos_triples):
        """Identify boundary entities for preservation."""
        relations = pos_triples[:, 1].unique()
        boundary_entities = set()

        for relation in relations:
            rel_mask = (pos_triples[:, 1] == relation)
            rel_triples = pos_triples[rel_mask]
            heads = rel_triples[:, 0].tolist()
            tails = rel_triples[:, 2].tolist()
            boundary_entities.update(heads + tails)

        return torch.tensor(list(boundary_entities), dtype=torch.long, device=self.args.device)

    def boundary_preservation_loss(self, preservation_entities):
        """Calculate boundary preservation loss."""
        current_embeddings = self.kge_model.ent_embeddings.weight

        if not hasattr(self, 'reference_model') or self.reference_model is None:
            reference_embeddings, _ = self.old_embeddings()
        else:
            reference_embeddings = self.kge_model.ent_embeddings.weight

        preservation_entities = preservation_entities.to(self.args.device)
        e = torch.index_select(current_embeddings, 0, preservation_entities)
        e_ref = torch.index_select(reference_embeddings, 0, preservation_entities)

        diffs = e - e_ref
        diff_norms = torch.norm(diffs, dim=1)

        mask_small = (diff_norms <= 1.0)
        mask_large = (diff_norms > 1.0)

        loss_small = 0.5 * torch.sum(diffs[mask_small] ** 2, dim=1)
        loss_large = diff_norms[mask_large] - 0.5

        total_loss = torch.zeros_like(diff_norms)
        total_loss[mask_small] = loss_small
        total_loss[mask_large] = loss_large

        return torch.mean(total_loss)

    def _create_matched_groups(self, pos_triples, neg_triples, pos_weights):
        """Create groups of positive and negative triples."""
        if self.args.grouping_strategy != 'batch':
            triple_groups = self.form_triple_groups(pos_triples, pos_weights)
            matched_groups = []

            for group_pos, group_weights in triple_groups:
                group_indices = torch.zeros(pos_triples.size(0), dtype=torch.bool, device=pos_triples.device)
                for pos_triple in group_pos:
                    matches = ((pos_triples == pos_triple).all(dim=1))
                    group_indices = group_indices | matches
                group_neg = neg_triples[group_indices]
                matched_groups.append((group_pos, group_neg, group_weights))
        else:
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
        """Efficient batched embedding lookup."""
        h = torch.index_select(ent_emb, 0, triples[:, 0])
        r = torch.index_select(rel_emb, 0, triples[:, 1])
        t = torch.index_select(ent_emb, 0, triples[:, 2])
        return h, r, t

    # Main loss function - updated to use INTUITOR approach
    def combined_loss(self, head, relation, tail, label, pos_triples, neg_triples, pos_weights=None):
        """Main combined loss function with INTUITOR integration."""
        return self.intuitor_combined_loss(head, relation, tail, label, pos_triples, neg_triples, pos_weights)