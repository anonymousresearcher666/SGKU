import time
import traceback
import torch
from torch.utils.data import DataLoader
from src.loading.loader import *
from src.utilities.utilities import should_pin_memory


class UnlTrainBatch():
    """ Standard  process """

    def __init__(self, args, kg) -> None:
        self.args = args
        self.kg = kg
        if self.args.begin_pretrain:
            self.dataset = TrainDataset(args, kg)
        elif self.args.begin_unleanring:
            if self.args.unlearning_method in ["pretrain","retrain", "tune"]:
                self.dataset = RetrainDataset(args, kg)
        else:
            raise ImportError

        self.data_loader = DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=int(self.args.batch_size),
            collate_fn=self.dataset.collate_fn,
            generator=torch.Generator().manual_seed(int(args.seed)),  # use seed generator
            pin_memory=should_pin_memory(self.args.device)
        )

    def process_epoch(self, model, optimizer):
        model.train()
        """ Start training """
        total_loss = 0.0
        epoch_start_time = time.time()
        max_batches = getattr(self.args, "max_train_batches", None)
        try:
            max_batches = int(max_batches) if max_batches is not None else None
        except (TypeError, ValueError):
            max_batches = None
        data_iter = self.data_loader
        show_progress = bool(getattr(self.args, "show_batch_progress", False))
        if getattr(self.args, "begin_pretrain", False) and show_progress:
            try:
                from tqdm import tqdm  # type: ignore
            except ImportError:  # pragma: no cover - tqdm optional
                tqdm = None
            if tqdm is not None:
                total = None
                try:
                    total = len(self.data_loader)
                except TypeError:
                    total = None
                epoch_id = int(getattr(self.args, "epoch", 0)) + 1
                data_iter = tqdm(
                    self.data_loader,
                    total=total,
                    desc=f"Epoch {epoch_id} train",
                    leave=False,
                )
        for b_id, batch in enumerate(data_iter):
            if max_batches is not None and b_id >= max_batches:
                break
            bh, br, bt, by = batch
            optimizer.zero_grad()
            batch_loss = model.loss(bh.to(self.args.device),
                                    br.to(self.args.device),
                                    bt.to(self.args.device),
                                    by.to(self.args.device) if by is not None else by).float()
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
        epoch_duration = time.time() - epoch_start_time
        return total_loss, epoch_duration


class SGKUBatching:
    """Optimized training processor for schema-guided unlearning with gradient-guided optimization"""
    def __init__(self, args, kg) -> None:
        """
        Initialize the training processor with schema guidance and selective forgetting capabilities.

        Args:
            args: Arguments with configuration
            kg: Knowledge graph instance
        """
        self.args = args
        self.kg = kg

        # Initialize caches and performance optimizations
        self._init_caches()

        # Setup schema store and efficient type lookups
        self._setup_schema_infrastructure()

        # Create dataset and data loader
        self._setup_data_loader()

        # Configure GRPO settings
        self._configure_grpo()

        # Precompute schema information where possible
        if hasattr(kg, 'relation_to_pattern') and kg.relation_to_pattern:
            self._precompute_schema_information()
        else:
            print("ERROR: KnowledgeGraph does not have relation_to_pattern mappings!")
            exit(1)

    def _init_caches(self):
        """Initialize all caches for maximum performance"""
        # Triple-pattern cache with higher initial capacity
        self._pattern_cache = {}

        # Weight cache with optimized structure
        self._weight_cache = {}

        # Entity and relation type caches
        self._entity_types = {}
        self._relation_types = {}

        # Schema info cache for batch processing
        self._batch_schema_cache = {}

        # Batch weight tensor cache (reused between similar batches)
        self._batch_weight_tensors = {}

        # Model-specific caches
        self._model_schema_cache = None

        # Relation pattern and importance caches
        self._relation_pattern_cache = {}
        self._pattern_importance_cache = {}

        # Batch counter for GRPO frequency
        self._batch_counter = 0

        # Stats tracking
        self._successful_batches = 0
        self._failed_batches = 0
        self._total_triples_processed = 0

    def _configure_grpo(self):
        """Configure gradient-guided optimization settings"""
        # Set GRPO parameters with defaults if not present
        self.args.use_gradient_guided_optimization = getattr(self.args, 'use_gradient_guided_optimization', False)
        self.args.grpo_frequency = getattr(self.args, 'grpo_frequency', 3)  # Apply GRPO every N batches
        self.args.grpo_batch_size = getattr(self.args, 'grpo_batch_size', self.args.batch_size)

        # Create dataset for forget/retain triples only when needed
        self._grpo_loaders_ready = False
        if self.args.use_gradient_guided_optimization or getattr(self.args, 'boundary_data', False):
            self._setup_grpo_data_loaders()
        else:
            # Ensure attributes exist even when loaders are skipped
            self.forget_dataset = None
            self.retain_dataset = None
            self.boundary_dataset = None
            self.forget_data_loader = None
            self.retain_data_loader = None
            self.boundary_data_loader = None

    def _setup_schema_infrastructure(self):
        """Set up schema handling infrastructure using ONLY precomputed KG data - no fallbacks allowed"""
        # STRICT CHECK: Validate ALL required precomputed structures exist
        if not hasattr(self.kg, 'schema_store') or not self.kg.schema_store:
            print("ERROR: KnowledgeGraph schema_store is missing! Run CreateSchema.py first.")
            exit(1)

        if not hasattr(self.kg, 'global_pattern_cache'):
            print("ERROR: KnowledgeGraph global_pattern_cache is missing!")
            print("This structure is required for efficient schema pattern lookup.")
            print("Run CreateSchema.py with --precompute-patterns flag.")
            exit(1)

        if not hasattr(self.kg, 'relation_to_pattern') or not self.kg.relation_to_pattern:
            print("ERROR: KnowledgeGraph relation_to_pattern mapping is missing!")
            print("This mapping is required for relation-based pattern lookup.")
            print("Run CreateSchema.py with --map-relations flag.")
            exit(1)

        # Verify id2relation mapping exists - this is CRUCIAL for reliable relation name lookup
        if not hasattr(self.kg, 'id2relation') or not self.kg.id2relation:
            print("ERROR: KnowledgeGraph id2relation mapping is missing or empty!")
            print("This mapping is required for looking up relation names from IDs.")
            print("Make sure your KnowledgeGraph has properly loaded relation mappings.")
            exit(1)

        # All required structures exist - use them directly
        self.schema_store = self.kg.schema_store
        self._pattern_cache = self.kg.global_pattern_cache
        self.relation_to_pattern = self.kg.relation_to_pattern

        # Use KG's weight cache if available, or initialize one
        if hasattr(self.kg, '_weight_cache'):
            self._weight_cache = self.kg._weight_cache
        else:
            # Create a new weight cache on the KG for future use
            self.kg._weight_cache = self._weight_cache

    def _setup_data_loader(self):
        """Set up optimized dataset and data loader"""
        # Create dataset
        self.dataset = UnifiedSchemaGuidedDataset(self.args, self.kg, DatasetType.MAIN)
        self.entity_distill_mask = self.dataset.entity_distill_mask
        # Optimize data loader for performance
        self.data_loader = DataLoader(
            self.dataset,
            shuffle=False,
            batch_size=int(self.args.batch_size),
            collate_fn=self.dataset.collate_fn,
            generator=torch.Generator().manual_seed(int(self.args.seed)),
            pin_memory=should_pin_memory(self.args.device),
            num_workers=0,  # Can't use workers due to pickling issues
            persistent_workers=False
        )
        # Track total number of batches for progress reporting
        self.total_batches = len(self.data_loader)
        # print(f"Created data loader with {self.total_batches} batches (batch size: {self.args.batch_size})")

    def _setup_grpo_data_loaders(self):
        """Set up specialized data loaders for gradient-guided optimization"""
        try:
            # Try to create forget and retain datasets
            self.forget_dataset = UnifiedSchemaGuidedDataset(self.args, self.kg, DatasetType.FORGET)
            self.retain_dataset = UnifiedSchemaGuidedDataset(self.args, self.kg, DatasetType.RETAIN)
            self.boundary_dataset = UnifiedSchemaGuidedDataset(self.args, self.kg, DatasetType.BOUNDARY)

            # Create data loaders with the same seed for reproducibility
            self.forget_data_loader = DataLoader(
                self.forget_dataset,
                shuffle=False,
                batch_size=self.args.grpo_batch_size,
                collate_fn=self.forget_dataset.collate_fn,
                generator=torch.Generator().manual_seed(int(self.args.seed)),
                pin_memory=should_pin_memory(self.args.device)
            )
            self.retain_data_loader = DataLoader(
                self.retain_dataset,
                shuffle=False,
                batch_size=self.args.grpo_batch_size,
                collate_fn=self.retain_dataset.collate_fn,
                generator=torch.Generator().manual_seed(int(self.args.seed)),
                pin_memory=should_pin_memory(self.args.device)
            )

            self.boundary_data_loader = DataLoader(
                self.boundary_dataset,
                shuffle=True,
                batch_size=self.args.grpo_batch_size,
                collate_fn=self.boundary_dataset.collate_fn,
                generator=torch.Generator().manual_seed(int(self.args.seed)),
                pin_memory=should_pin_memory(self.args.device)
            )

            # Flag that GRPO data loaders are ready
            self._grpo_loaders_ready = True

        except (ImportError, AttributeError) as e:
            # Fall back to using schema-guided dataset if specialized datasets aren't available
            print(f"Warning: Could not create specialized GRPO datasets ({str(e)})")
            print("Falling back to using schema-guided dataset for GRPO")

            exit(1)

    def _precompute_schema_information(self):
        """Precompute and cache schema patterns and importance values"""
        if not hasattr(self.kg, 'id2relation') or not self.kg.id2relation:
            print("ERROR: KnowledgeGraph does not have id2relation mappings!")
            exit(1)

        start_time = time.time()
        rel_count = 0

        # Cache relation patterns and importance values
        for rel_id, rel_type in self.kg.id2relation.items():
            if rel_type in self.kg.relation_to_pattern:
                pattern = self.kg.relation_to_pattern[rel_type]
                self._relation_pattern_cache[rel_id] = pattern
                # Cache importance values
                if pattern in self.schema_store and 'importance' in self.schema_store[pattern]:
                    self._pattern_importance_cache[pattern] = self.schema_store[pattern]['importance']
                rel_count += 1
        duration = time.time() - start_time
        print(f"Precomputed schema information for {rel_count} relations in {duration:.2f} seconds")

    def process_epoch(self, model, optimizer):
        """Process training epoch with schema-guided loss and gradient-guided optimization.

        Args:
            model: The SchemaGRPO model instance
            optimizer: PyTorch optimizer

        Returns:
            float: Total loss for the epoch
        """
        device = self.args.device
        model.train()

        # Set entity distill mask on model if available
        if hasattr(self, 'entity_distill_mask') and self.entity_distill_mask is not None:
            model.entity_distill_mask = self.entity_distill_mask

        total_loss = 0.0
        grpo_applications = 0
        self._batch_counter = 0
        self._successful_batches = 0
        self._failed_batches = 0
        self._total_triples_processed = 0

        # Setup embeddings and caches
        self._setup_embeddings_and_caches(model)

        # Ensure model has optimizer reference for GRPO
        if self.args.use_gradient_guided_optimization and not hasattr(model, 'optimizer'):
            model.optimizer = optimizer

        # Print training start info
        print(f"Model is on device: {device}")
        print(f"Processing {len(self.dataset)} triples in {self.total_batches} batches")
        print(f"Batch size: {self.args.batch_size}, use_gradient_guided_optimization enabled: {self.args.use_gradient_guided_optimization}")

        if self.total_batches == 0:
            print("No training batches available; skipping epoch.")
            return total_loss, 0.0

        # Track progress
        epoch_start_time = time.time()
        progress_start = epoch_start_time
        progress_bar_length = 30

        max_batches = getattr(self.args, "max_train_batches", None)
        try:
            max_batches = int(max_batches) if max_batches is not None else None
        except (TypeError, ValueError):
            max_batches = None

        # Process batches
        for b_id, batch in enumerate(self.data_loader):
            if max_batches is not None and b_id >= max_batches:
                break
            self._batch_counter += 1
            current_batch = b_id + 1

            # Update progress bar
            progress = current_batch / self.total_batches
            bar_filled = int(progress_bar_length * progress)
            bar = '█' * bar_filled + '░' * (progress_bar_length - bar_filled)

            # Calculate ETA
            elapsed = time.time() - progress_start
            if current_batch > 1:
                eta = elapsed / (current_batch - 1) * (self.total_batches - current_batch + 1)
                eta_str = f"ETA: {eta:.1f}s"
            else:
                eta_str = "ETA: ..."

            # Print progress
            print(f"\rBatch: {current_batch}/{self.total_batches} [{bar}] {progress:.1%} {eta_str}", end="", flush=True)

            try:
                # Prepare batch data
                bh, br, bt, by, pos_facts, neg_facts = self._prepare_batch_tensors(batch, device)
                # Update triple count
                batch_size = bh.size(0)
                self._total_triples_processed += batch_size

                # Get schema weights for positive facts (only if needed)
                pos_schema_weights = self._extract_patterns_and_weights(pos_facts[:, 1])

                # Forward pass and loss calculation
                optimizer.zero_grad()
                if hasattr(model, 'intuitor_combined_loss'):
                    loss_fn = model.intuitor_combined_loss
                else:
                    loss_fn = model.combined_loss
                batch_loss = loss_fn(
                    bh, br, bt, by, pos_facts, neg_facts, pos_schema_weights
                ).float()

                # Backward pass (if loss is non-zero)
                if batch_loss > 1e-8:
                    batch_loss.backward()
                    optimizer.step()
                    total_loss += batch_loss.item()
                    self._successful_batches += 1

                # Apply gradient-guided optimization periodically if enabled
                if (self.args.use_gradient_guided_optimization and
                        self._batch_counter % self.args.grpo_frequency == 0):
                    # Update progress to show GRPO
                    print(f"\rBatch: {current_batch}/{self.total_batches} [{bar}] {progress:.1%} Applying GRPO...",
                          end="", flush=True)

                    # Apply GRPO and get metrics (not losses to be added)
                    grpo_metrics = self._apply_gradient_guided_optimization(model)

                    # Ensure grpo_metrics is a dictionary
                    if not isinstance(grpo_metrics, dict):
                        grpo_metrics = {'projection_magnitude': grpo_metrics, 'projection_needed': True}

                    # Now safely use get()
                    if grpo_metrics.get('projection_needed', False):
                        # The projection is already applied within gradient_guided_optimization_step
                        #
                        # Instead, we just need to track the application:
                        grpo_applications += 1

                    # Track metrics separately for reporting
                    grpo_effect = grpo_metrics.get('projection_magnitude', 0.0)


            except Exception as e:
                self._log_batch_error(b_id, e)
                self._failed_batches += 1
                continue

        # Print a new line after progress bar
        print()  # New line after main data loader progress

        # Process boundary dataset if enabled
        if self.args.boundary_data and self.boundary_data_loader is not None:
            print(" ------------------ BOUNDARY ----------------------")
            boundary_batches = len(self.boundary_data_loader)
            for b_id, batch in enumerate(self.boundary_data_loader):
                # Update boundary progress
                boundary_progress = (b_id + 1) / boundary_batches
                boundary_bar_filled = int(progress_bar_length * boundary_progress)
                boundary_bar = '█' * boundary_bar_filled + '░' * (progress_bar_length - boundary_bar_filled)
                print(f"\rBoundary: {b_id + 1}/{boundary_batches} [{boundary_bar}] {boundary_progress:.1%}",
                      end="", flush=True)

                try:
                    # Batch format: (h, r, t, label, importance)
                    bh, br, bt, by, importance = batch

                    # Skip batches with no important triples
                    if importance.max().item() < 1e-6:
                        continue

                    # Move tensors to device
                    bh = bh.to(self.args.device)
                    br = br.to(self.args.device)
                    bt = bt.to(self.args.device)
                    by = by.to(self.args.device) if by is not None else by
                    importance = importance.to(self.args.device)

                    # Zero gradients before boundary optimization
                    optimizer.zero_grad()

                    # Get batch loss
                    batch_loss = model.kge_model.loss(bh, br, bt, by).float()

                    # Apply importance weighting
                    if batch_loss.dim() > 0 and batch_loss.size(0) == bh.size(0):
                        # Element-wise weighting for per-sample losses
                        weighted_loss = (batch_loss * importance).mean()
                    else:
                        # Scale aggregated loss by mean importance
                        weighted_loss = batch_loss * importance.mean()

                    # Only proceed if loss is significant
                    if weighted_loss > 1e-8:
                        weighted_loss.backward()
                        optimizer.step()
                        total_loss += weighted_loss.item()

                except Exception as e:
                    self._log_batch_error(f"boundary_{b_id}", e)
                    continue

            # Print new line after boundary progress
            print()

        # Update pattern cache
        self._update_global_caches()

        # Calculate total time
        total_time = time.time() - epoch_start_time
        avg_time_per_batch = total_time / max(1, self._successful_batches)

        # Log summary
        print("====================================")
        print(f"Total time: {total_time:.2f} seconds")
        print(
            f"Successful batches: {self._successful_batches}/{self.total_batches} ({self._successful_batches / self.total_batches:.1%})")
        print(f"Failed batches: {self._failed_batches}")
        print(f"Triples processed: {self._total_triples_processed}")
        print(f"Average time per batch: {avg_time_per_batch:.4f} seconds")
        print(f"Average loss: {total_loss / max(1, self._successful_batches):.4f}")
        if self.args.use_gradient_guided_optimization:
            self._log_epoch_summary(grpo_applications)
        print("====================================")
        return total_loss, total_time

    def _setup_embeddings_and_caches(self, model):
        """Setup model embeddings and synchronize caches"""
        # Save model embeddings as reference for GRPO/distillation
        if hasattr(model, 'save_embeddings'):
            model.save_embeddings()

        # Share pattern cache
        if hasattr(model, '_pattern_cache'):
            if hasattr(self.kg, 'global_pattern_cache'):
                model._pattern_cache = self.kg.global_pattern_cache
                self._pattern_cache = self.kg.global_pattern_cache
            elif not hasattr(self, '_pattern_cache') or len(model._pattern_cache) > len(
                    getattr(self, '_pattern_cache', {})):
                self._pattern_cache = model._pattern_cache
            else:
                model._pattern_cache = self._pattern_cache

        # Share weight cache
        if hasattr(self.kg, '_weight_cache'):
            self._weight_cache = self.kg._weight_cache
            if hasattr(model, '_weight_cache'):
                model._weight_cache = self.kg._weight_cache

        # Process entity mask
        if self.entity_distill_mask is not None:
            if not isinstance(self.entity_distill_mask, torch.Tensor):
                self.entity_distill_mask = torch.tensor(self.entity_distill_mask, device=self.args.device)
            else:
                self.entity_distill_mask = self.entity_distill_mask.to(self.args.device)

    def _prepare_batch_tensors(self, batch, device):
        """Extract and prepare tensors from batch"""
        if len(batch) != 6:
            raise ValueError(f"Expected batch with 6 elements, got {len(batch)}")

        # Unpack batch
        bh, br, bt, by, pos_facts, neg_facts = batch
        # Process entity tensors - ensure 1D
        bh = bh.squeeze().to(device)
        br = br.squeeze().to(device)
        bt = bt.squeeze().to(device)
        # Process label tensor
        by = by.to(device)
        # Fix shape mismatch if needed
        if by.shape[0] != bh.shape[0]:
            print(f"WARNING: Label tensor shape mismatch: by.shape={by.shape}, bh.shape={bh.shape}")
            if len(by) % len(bh) == 0:
                repeat_factor = len(by) // len(bh)
                by = by.reshape(len(bh), repeat_factor)
                if by.dim() > 1:
                    by = by[:, 0]
            else:
                by = by[:len(bh)]

        # Process fact tensors
        pos_facts = pos_facts.to(device)
        neg_facts = neg_facts.to(device)

        return bh, br, bt, by, pos_facts, neg_facts

    def _extract_patterns_and_weights(self, br):
        """
        Fully vectorized function for pattern and weight extraction
        that leverages pre-computed caches with minimal processing
        and optimized CPU-GPU transfers
        """
        # Pre-allocate outputs
        batch_size = br.size(0)
        weights = torch.ones(batch_size, device=br.device)

        # Move relation IDs to CPU once (outside the loop)
        rel_ids = br.cpu().numpy()

        # Vectorized approach: Create a mapping dictionary for this batch
        # This handles all relation IDs that are in the cache
        cached_indices = {}
        uncached_rel_ids = []
        uncached_positions = []

        # First pass: identify cached and uncached relations
        for i, rel_id in enumerate(rel_ids):
            if rel_id in self._relation_pattern_cache:
                pattern = self._relation_pattern_cache[rel_id]
                cached_indices[i] = pattern
            else:
                uncached_rel_ids.append(rel_id)
                uncached_positions.append(i)

        # Process all patterns (both cached and newly computed)
        batch_patterns = [None] * batch_size  # Pre-allocate list with correct size

        # Fill in patterns from cache
        for i, pattern in cached_indices.items():
            batch_patterns[i] = pattern
            if pattern in self._pattern_importance_cache:
                weights[i] = self._pattern_importance_cache[pattern]

        # Process uncached relation IDs (only if there are any)
        if uncached_rel_ids:
            for j, rel_id in enumerate(uncached_rel_ids):
                i = uncached_positions[j]  # Original position in batch

                # Get relation name and create default pattern
                rel_name = self.kg.id2relation.get(rel_id, f"relation_{rel_id}")
                pattern = ("Entity", rel_name, "Entity")

                # Cache for future use
                self._relation_pattern_cache[rel_id] = pattern

                # Store in output
                batch_patterns[i] = pattern

                # Weight lookup
                if pattern in self._pattern_importance_cache:
                    weights[i] = self._pattern_importance_cache[pattern]

        return weights

    def _compute_loss(self, model, bh, br, bt, by, pos_facts, neg_facts, pos_schema_weights):
        """Compute loss with schema weights"""
        try:
            total_loss = model.combined_loss(
                bh, br, bt, by, pos_facts, neg_facts, pos_schema_weights
            ).float()

            return total_loss

        except Exception as e:
            print(f"FATAL ERROR in loss computation: {str(e)}")
            print("Input tensor details:")
            print(f"  bh: shape={bh.shape}, type={bh.dtype}, device={bh.device}")
            print(f"  br: shape={br.shape}, type={br.dtype}, device={br.device}")
            print(f"  bt: shape={bt.shape}, type={bt.dtype}, device={bt.device}")
            print(f"  by: shape={by.shape}, type={by.dtype}, device={by.device}")
            print(f"  pos_facts: shape={pos_facts.shape}, type={pos_facts.dtype}, device={pos_facts.device}")
            print(f"  neg_facts: shape={neg_facts.shape}, type={neg_facts.dtype}, device={neg_facts.device}")
            print(
                f"  schema_weights: shape={pos_schema_weights.shape}, type={pos_schema_weights.dtype}, device={pos_schema_weights.device}")
            print(f"  by has NaN: {torch.isnan(by).any()}, has Inf: {torch.isinf(by).any()}")
            print(
                f"  schema_weights has NaN: {torch.isnan(pos_schema_weights).any()}, has Inf: {torch.isinf(pos_schema_weights).any()}")
            raise  # Re-raise exception for proper debugging

    def _apply_gradient_guided_optimization(self, model):
        """Apply gradient-guided optimization for selective forgetting.

        Args:
            model: The SGKU model

        Returns:
            Tensor: Forget loss value
        """
        try:
            # Get batch of triples to forget
            forget_batch = self._get_next_batch(self.forget_data_loader)
            if not forget_batch:
                return 0.0
            forget_triples = self._prepare_triple_tensor(forget_batch)

            # Get batch of triples to retain
            retain_batch = self._get_next_batch(self.retain_data_loader)
            if not retain_batch:
                return 0.0
            retain_triples = self._prepare_triple_tensor(retain_batch)

            # Get schema weights
            retain_schema_weights = self._extract_patterns_and_weights(retain_triples[:, 1])

            # Apply optimization step
            result = model.gradient_guided_optimization_step(
                forget_triples=forget_triples,
                retain_triples=retain_triples,
                schema_weights=retain_schema_weights
            )

            return result.get('forget_loss', 0.0)

        except Exception as e:
            self._log_grpo_error(e)
            return 0.0

    def _prepare_triple_tensor(self, batch):
        """Convert a batch to a triple tensor.

        Args:
            batch: Tuple of (head, relation, tail, label) tensors

        Returns:
            Tensor: Combined triple tensor
        """
        h, r, t, _ = batch
        return torch.stack([
            h.squeeze(), r.squeeze(), t.squeeze()
        ], dim=1).to(self.args.device)

    def _get_next_batch(self, data_loader):
        """Get next batch safely from a data loader.

        Args:
            data_loader: PyTorch DataLoader

        Returns:
            tuple or None: Batch data or None if invalid
        """
        try:
            batch = next(iter(data_loader))
            if len(batch) < 4:
                return None
            return batch
        except Exception:
            return None

    def _log_batch_error(self, batch_id, error):
        """Log batch processing error."""
        print(f"\nERROR processing batch {batch_id}: {str(error)}")
        traceback.print_exc()

    def _log_grpo_error(self, error):
        """Log GRPO-specific error."""
        print(f"\nERROR in gradient-guided optimization: {str(error)}")
        traceback.print_exc()

    def _log_epoch_summary(self, grpo_applications):
        """Log epoch summary."""
        print(f"Applied gradient-guided optimization {grpo_applications} times")
        if hasattr(self, 'forget_dataset') and hasattr(self, 'retain_dataset'):
            print(
                f"GRPO datasets: {len(self.forget_dataset)} forget triples, {len(self.retain_dataset)} retain triples")

    def _update_global_caches(self):
        """Update global pattern caches."""
        if hasattr(self.kg, 'global_pattern_cache') and hasattr(self, '_pattern_cache'):
            self.kg.global_pattern_cache.update(self._pattern_cache)


class SGKUPaperBatching:
    """Paper-aligned SGKU training loop (retain-vs-forget GRPO batches).

    This replaces the legacy MAIN-dataset + heuristic GRPO logic by directly optimizing
    the paper objective on mixed retain/forget minibatches, with:
      - group-softmax policies over standardized logits,
      - PPO-style clipping + KL penalty vs reference (theta_old),
      - group-level weights computed once per timestep,
      - boundary Huber preservation vs initial embeddings,
      - periodic conflict-aware projection step.
    """

    def __init__(self, args, kg) -> None:
        self.args = args
        self.kg = kg

        # Datasets for the current timestep (facts already include inverse triples).
        self.forget_dataset = UnifiedSchemaGuidedDataset(self.args, self.kg, DatasetType.FORGET)
        self.retain_dataset = UnifiedSchemaGuidedDataset(self.args, self.kg, DatasetType.RETAIN)

        self.forget_loader = DataLoader(
            self.forget_dataset,
            shuffle=True,
            batch_size=int(self.args.batch_size),
            collate_fn=self.forget_dataset.collate_fn,
            generator=torch.Generator().manual_seed(int(self.args.seed)),
            pin_memory=should_pin_memory(self.args.device),
            num_workers=0,
            persistent_workers=False,
        )
        self.retain_loader = DataLoader(
            self.retain_dataset,
            shuffle=True,
            batch_size=int(self.args.batch_size),
            collate_fn=self.retain_dataset.collate_fn,
            generator=torch.Generator().manual_seed(int(self.args.seed)),
            pin_memory=should_pin_memory(self.args.device),
            num_workers=0,
            persistent_workers=False,
        )

        self.boundary_entities = self._compute_boundary_entities_tensor()
        self.grouping_strategy = self._normalize_grouping_strategy()
        self._init_grouping_state()
        self.group_weight_tensor = self._compute_group_weight_tensor()

        self._initialized_model_state = False
        self._global_step = 0
        self._last_timestep = None
        self._kl_ema = None

    def _compute_boundary_entities_tensor(self) -> torch.Tensor:
        """Boundary entities appear in both forget and retain contexts for this timestep."""
        timestep_idx = int(getattr(self.args, "timestep", 0) or 0)
        timestep = self.kg.timesteps[timestep_idx]
        forget_entities = set()
        retain_entities = set()
        for h, r, t in getattr(timestep, "forgotten_triples", []):
            forget_entities.add(int(h))
            forget_entities.add(int(t))
        for h, r, t in getattr(timestep, "retain_triples", []):
            retain_entities.add(int(h))
            retain_entities.add(int(t))
        boundary = sorted(forget_entities.intersection(retain_entities))
        if not boundary:
            return torch.empty((0,), dtype=torch.long)
        return torch.tensor(boundary, dtype=torch.long)

    def _normalize_grouping_strategy(self) -> str:
        strategy = str(getattr(self.args, "grouping_strategy", "relation") or "relation").lower()
        if strategy in {"rel", "relation", "relations"}:
            return "relation"
        if strategy in {"typed", "typed_relation", "typed-relations", "schema", "pattern", "patterns"}:
            return "typed"
        return strategy

    def _init_grouping_state(self) -> None:
        self._group_id_cache = {}
        self._group_key_by_id = {}
        self._next_group_id = 0

        # Precompute entity type ids for typed grouping.
        self._entity_type_ids = None
        self._type_name_by_id = {}
        self._rel_key_id_by_rel = None
        self._rel_key_name_by_id = {}

        if self.grouping_strategy != "typed":
            return

        ent_num = int(getattr(self.kg, "ent_num", 0) or 0)
        entity_types = getattr(self.kg, "entity_types", {}) or {}
        type_vocab = {"Entity": 0}
        type_name_by_id = {0: "Entity"}
        ent_type_ids = torch.zeros((ent_num,), dtype=torch.long)
        for ent_id in range(ent_num):
            types = entity_types.get(ent_id)
            if isinstance(types, (set, list, tuple)):
                type_name = sorted(list(types))[0] if types else "Entity"
            elif types:
                type_name = str(types)
            else:
                type_name = "Entity"
            if type_name not in type_vocab:
                type_id = len(type_vocab)
                type_vocab[type_name] = type_id
                type_name_by_id[type_id] = type_name
            ent_type_ids[ent_id] = type_vocab[type_name]
        self._entity_type_ids = ent_type_ids
        self._type_name_by_id = type_name_by_id

        rel_num = int(getattr(self.kg, "rel_num", 0) or 0)
        id2relation = getattr(self.kg, "id2relation", {}) or {}
        rel_key_vocab = {}
        rel_key_name_by_id = {}
        rel_key_ids = torch.zeros((rel_num,), dtype=torch.long)
        for rel_id in range(rel_num):
            rel_name = self._relation_name_for_grouping(rel_id, id2relation)
            if rel_name not in rel_key_vocab:
                key_id = len(rel_key_vocab)
                rel_key_vocab[rel_name] = key_id
                rel_key_name_by_id[key_id] = rel_name
            rel_key_ids[rel_id] = rel_key_vocab[rel_name]
        self._rel_key_id_by_rel = rel_key_ids
        self._rel_key_name_by_id = rel_key_name_by_id

    def _relation_name_for_grouping(self, rel_id: int, id2relation: dict) -> str:
        if rel_id in id2relation:
            return id2relation[rel_id]
        # Fall back to even/odd inverse pairing if possible.
        if rel_id % 2 == 1 and (rel_id - 1) in id2relation:
            return id2relation[rel_id - 1]
        if rel_id % 2 == 0 and (rel_id + 1) in id2relation:
            return id2relation[rel_id + 1]
        return f"relation_{rel_id}"

    def _group_id_for_triple(self, h: int, r: int, t: int) -> int:
        if self.grouping_strategy == "relation":
            return int(r)

        if self._entity_type_ids is None or self._rel_key_id_by_rel is None:
            return int(r)

        ent_num = int(self._entity_type_ids.size(0))
        rel_num = int(self._rel_key_id_by_rel.size(0))
        h_type = int(self._entity_type_ids[h]) if 0 <= h < ent_num else 0
        t_type = int(self._entity_type_ids[t]) if 0 <= t < ent_num else 0
        rel_key = int(self._rel_key_id_by_rel[r]) if 0 <= r < rel_num else int(r)

        key = (h_type, rel_key, t_type)
        gid = self._group_id_cache.get(key)
        if gid is None:
            gid = self._next_group_id
            self._group_id_cache[key] = gid
            self._group_key_by_id[gid] = key
            self._next_group_id += 1
        return gid

    def _group_ids_for_triples(self, triples: torch.Tensor) -> torch.Tensor:
        if self.grouping_strategy == "relation":
            return triples[:, 1].long()
        triples_cpu = triples.detach().cpu().numpy()
        group_ids = [self._group_id_for_triple(int(h), int(r), int(t)) for h, r, t in triples_cpu]
        return torch.tensor(group_ids, dtype=torch.long)

    def _compute_relation_group_weights(self) -> dict:
        """Compute group weights w_g in (0,1] for relation-level groups.

        Modes:
          - retention_coverage (default): paper data-driven weights
          - expert: use schema_store importance per relation pattern when available
          - uniform: w_g = 1
        """
        mode = str(getattr(self.args, "group_weight_mode", "retention_coverage") or "retention_coverage").lower()
        if mode == "uniform":
            return {}
        if mode == "expert":
            weights = {}
            schema_store = getattr(self.kg, "schema_store", None) or {}
            id2relation = getattr(self.kg, "id2relation", None) or {}
            relation_to_pattern = getattr(self.kg, "relation_to_pattern", None) or {}
            for rel_id, rel_name in id2relation.items():
                try:
                    pattern = relation_to_pattern.get(rel_name)
                    if pattern and pattern in schema_store:
                        weights[int(rel_id)] = float(schema_store[pattern].get("importance", 1.0))
                    else:
                        weights[int(rel_id)] = 1.0
                except Exception:
                    weights[int(rel_id)] = 1.0
            max_w = max(weights.values()) if weights else 1.0
            if max_w <= 0:
                return {k: 1.0 for k in weights.keys()}
            return {k: float(v) / float(max_w) for k, v in weights.items()}

        alpha = float(getattr(self.args, "group_weight_alpha", 0.5))
        alpha = max(0.0, min(1.0, alpha))

        retain_counts = {}
        total_counts = {}

        # Use dataset facts (includes inverse triples, matching training batches).
        for entry in getattr(self.retain_dataset, "facts", []) or []:
            h, r, t = entry.get("fact", (None, None, None))
            if r is None:
                continue
            r = int(r)
            retain_counts[r] = retain_counts.get(r, 0) + 1
            total_counts[r] = total_counts.get(r, 0) + 1
        for entry in getattr(self.forget_dataset, "facts", []) or []:
            h, r, t = entry.get("fact", (None, None, None))
            if r is None:
                continue
            r = int(r)
            total_counts[r] = total_counts.get(r, 0) + 1

        total_triples = float(sum(total_counts.values()) or 1.0)
        weights = {}
        for rel, total in total_counts.items():
            retain = float(retain_counts.get(rel, 0))
            total = float(total)
            retain_density = retain / max(1.0, total)
            coverage = total / total_triples
            w = alpha * retain_density + (1.0 - alpha) * coverage
            weights[int(rel)] = w

        max_w = max(weights.values()) if weights else 1.0
        if max_w <= 0:
            return {k: 1.0 for k in weights.keys()}
        return {k: float(v) / float(max_w) for k, v in weights.items()}

    def _build_relation_weight_tensor(self, weights_by_relation: dict) -> torch.Tensor:
        rel_num = int(getattr(self.kg, "rel_num", 0) or 0)
        if rel_num <= 0:
            return torch.empty((0,), dtype=torch.float)
        weights = torch.ones((rel_num,), dtype=torch.float)
        for rel, weight in (weights_by_relation or {}).items():
            try:
                weights[int(rel)] = float(weight)
            except Exception:
                continue
        return weights

    def _compute_pattern_group_weights(self) -> torch.Tensor:
        mode = str(getattr(self.args, "group_weight_mode", "retention_coverage") or "retention_coverage").lower()
        alpha = float(getattr(self.args, "group_weight_alpha", 0.5))
        alpha = max(0.0, min(1.0, alpha))

        retain_counts = {}
        total_counts = {}

        for entry in getattr(self.retain_dataset, "facts", []) or []:
            h, r, t = entry.get("fact", (None, None, None))
            if h is None or r is None or t is None:
                continue
            gid = self._group_id_for_triple(int(h), int(r), int(t))
            retain_counts[gid] = retain_counts.get(gid, 0) + 1
            total_counts[gid] = total_counts.get(gid, 0) + 1

        for entry in getattr(self.forget_dataset, "facts", []) or []:
            h, r, t = entry.get("fact", (None, None, None))
            if h is None or r is None or t is None:
                continue
            gid = self._group_id_for_triple(int(h), int(r), int(t))
            total_counts[gid] = total_counts.get(gid, 0) + 1

        group_count = max(self._next_group_id, 1)

        if mode == "uniform":
            return torch.ones((group_count,), dtype=torch.float)

        if mode == "expert":
            schema_store = getattr(self.kg, "schema_store", None) or {}
            weights = {}
            for gid, key in self._group_key_by_id.items():
                h_type_id, rel_key_id, t_type_id = key
                h_type = self._type_name_by_id.get(h_type_id, "Entity")
                rel_name = self._rel_key_name_by_id.get(rel_key_id, f"relation_{rel_key_id}")
                t_type = self._type_name_by_id.get(t_type_id, "Entity")
                pattern = (h_type, rel_name, t_type)
                importance = 1.0
                if pattern in schema_store:
                    try:
                        importance = float(schema_store[pattern].get("importance", 1.0))
                    except Exception:
                        importance = 1.0
                weights[int(gid)] = importance
            max_w = max(weights.values()) if weights else 1.0
            max_w = max(max_w, 1e-8)
            out = torch.ones((group_count,), dtype=torch.float)
            for gid, weight in weights.items():
                out[int(gid)] = float(weight) / max_w
            return out

        total_triples = float(sum(total_counts.values()) or 1.0)
        weights = {}
        for gid, total in total_counts.items():
            retain = float(retain_counts.get(gid, 0))
            total = float(total)
            retain_density = retain / max(1.0, total)
            coverage = total / total_triples
            weights[int(gid)] = alpha * retain_density + (1.0 - alpha) * coverage

        max_w = max(weights.values()) if weights else 1.0
        max_w = max(max_w, 1e-8)
        out = torch.ones((group_count,), dtype=torch.float)
        for gid, weight in weights.items():
            out[int(gid)] = float(weight) / max_w
        return out

    def _compute_group_weight_tensor(self) -> torch.Tensor:
        if self.grouping_strategy == "relation":
            weights = self._compute_relation_group_weights()
            return self._build_relation_weight_tensor(weights)
        return self._compute_pattern_group_weights()

    def _ensure_group_weight_tensor(self, group_ids: torch.Tensor) -> None:
        if group_ids is None or group_ids.numel() == 0:
            return
        max_gid = int(group_ids.max().item())
        if max_gid < 0:
            return
        if self.group_weight_tensor is None or self.group_weight_tensor.numel() == 0:
            self.group_weight_tensor = torch.ones((max_gid + 1,), dtype=torch.float)
            return
        if max_gid >= self.group_weight_tensor.numel():
            extra = max_gid + 1 - self.group_weight_tensor.numel()
            pad = torch.ones((extra,), dtype=self.group_weight_tensor.dtype)
            self.group_weight_tensor = torch.cat([self.group_weight_tensor, pad], dim=0)

    def _ensure_model_initialized(self, model) -> None:
        if self._initialized_model_state:
            return
        # Optimizer reference is used by the periodic projection step.
        if not hasattr(model, "optimizer"):
            model.optimizer = None
        # Save initial and reference embeddings once per timestep.
        if hasattr(model, "save_initial_embeddings"):
            model.save_initial_embeddings()
        if hasattr(model, "refresh_reference_embeddings"):
            model.refresh_reference_embeddings()
        if hasattr(model, "set_boundary_entities"):
            model.set_boundary_entities(self.boundary_entities)
        self._initialized_model_state = True

    def process_epoch(self, model, optimizer):
        device = self.args.device
        model.train()

        cur_ts = int(getattr(self.args, "timestep", 0) or 0)
        if self._last_timestep is None or self._last_timestep != cur_ts:
            # Refresh per-timestep anchors and boundary sets.
            self._initialized_model_state = False
            self._last_timestep = cur_ts
            self._kl_ema = None

        self._ensure_model_initialized(model)
        model.optimizer = optimizer

        steps_per_epoch = getattr(self.args, "steps_per_epoch", None)
        if steps_per_epoch is None:
            steps_per_epoch = min(len(self.forget_loader), len(self.retain_loader))
        max_batches = getattr(self.args, "max_train_batches", None)
        try:
            max_batches = int(max_batches) if max_batches is not None else None
        except (TypeError, ValueError):
            max_batches = None
        if max_batches is not None and max_batches > 0:
            steps_per_epoch = min(int(steps_per_epoch or 0), max_batches)
        steps_per_epoch = int(steps_per_epoch or 0)
        if steps_per_epoch <= 0:
            return 0.0, 0.0

        clip_every_k = int(getattr(self.args, "projection_every_k", 20) or 20)
        ref_refresh_m = int(getattr(self.args, "reference_refresh_m", 10) or 10)
        kl_tau = getattr(self.args, "tau_kl", None)
        try:
            kl_tau = None if kl_tau is None else float(kl_tau)
        except (TypeError, ValueError):
            kl_tau = None
        kl_ema_alpha = float(getattr(self.args, "kl_refresh_ema_alpha", 0.1) or 0.1)
        kl_ema_alpha = max(0.0, min(1.0, kl_ema_alpha))

        total_loss = 0.0
        start = time.time()

        try:
            from tqdm import tqdm  # type: ignore
        except ImportError:  # pragma: no cover
            tqdm = None

        retain_iter = iter(self.retain_loader)
        forget_iter = iter(self.forget_loader)
        iterator = range(steps_per_epoch)
        if tqdm is not None and bool(getattr(self.args, "show_batch_progress", True)):
            epoch_id = int(getattr(self.args, "epoch", 0)) + 1
            method_name = getattr(self.args, "unlearning_method", getattr(self.args, "method", "SGKU"))
            iterator = tqdm(iterator, total=steps_per_epoch, desc=f"Epoch {epoch_id} {method_name}", leave=False)

        for _ in iterator:
            try:
                rh, rr, rt, _ = next(retain_iter)
            except StopIteration:
                retain_iter = iter(self.retain_loader)
                rh, rr, rt, _ = next(retain_iter)
            try:
                fh, fr, ft, _ = next(forget_iter)
            except StopIteration:
                forget_iter = iter(self.forget_loader)
                fh, fr, ft, _ = next(forget_iter)

            retain_triples = torch.stack([rh.squeeze(), rr.squeeze(), rt.squeeze()], dim=1)
            forget_triples = torch.stack([fh.squeeze(), fr.squeeze(), ft.squeeze()], dim=1)

            retain_group_ids = self._group_ids_for_triples(retain_triples)
            forget_group_ids = self._group_ids_for_triples(forget_triples)
            self._ensure_group_weight_tensor(torch.cat([retain_group_ids, forget_group_ids], dim=0))

            retain_triples = retain_triples.to(device)
            forget_triples = forget_triples.to(device)
            retain_group_ids = retain_group_ids.to(device)
            forget_group_ids = forget_group_ids.to(device)

            optimizer.zero_grad()
            loss = model.paper_total_loss(
                retain_triples=retain_triples,
                forget_triples=forget_triples,
                group_ids_retain=retain_group_ids,
                group_ids_forget=forget_group_ids,
                group_weight_tensor=self.group_weight_tensor.to(device),
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu().item())

            self._global_step += 1

            # Periodic conflict-aware projection step (paper: every K steps).
            if clip_every_k > 0 and (self._global_step % clip_every_k == 0):
                if hasattr(model, "conflict_aware_projection_step"):
                    model.conflict_aware_projection_step(
                        forget_triples=forget_triples,
                        retain_triples=retain_triples,
                        group_ids_forget=forget_group_ids,
                        group_ids_retain=retain_group_ids,
                        group_weight_tensor=self.group_weight_tensor.to(device),
                    )

            # Reference refresh (paper: every M steps).
            refresh_by_period = ref_refresh_m > 0 and (self._global_step % ref_refresh_m == 0)
            refresh_by_kl = False
            cur_kl = getattr(model, "last_batch_kl", None)
            if cur_kl is not None:
                try:
                    cur_kl = float(cur_kl)
                    if self._kl_ema is None:
                        self._kl_ema = cur_kl
                    else:
                        self._kl_ema = (1.0 - kl_ema_alpha) * float(self._kl_ema) + kl_ema_alpha * cur_kl
                    if kl_tau is not None and self._kl_ema > kl_tau:
                        refresh_by_kl = True
                except (TypeError, ValueError):
                    pass

            if refresh_by_period or refresh_by_kl:
                if hasattr(model, "refresh_reference_embeddings"):
                    model.refresh_reference_embeddings()
                # Reset EMA after a KL-triggered refresh to avoid immediate retrigger on stale average.
                if refresh_by_kl and cur_kl is not None:
                    self._kl_ema = cur_kl

        duration = time.time() - start
        return total_loss, duration


class DBatching():
    """
    test
    """

    def __init__(self, args, kg) -> None:
        self.args = args
        self.kg = kg
        self.batch_size = 100
        self.dataset = TestDataset(args, kg)
        self.data_loader = DataLoader(
            self.dataset,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=self.dataset.collate_fn,
            generator=torch.Generator().manual_seed(int(args.seed)),
            pin_memory=should_pin_memory(self.args.device)
        )

    def process_epoch(self, model):
        model.eval()
        num = 0
        results = {}
        start_time = time.time()
        data_iter = self.data_loader
        show_progress = bool(getattr(self.args, "show_eval_progress", False))
        if getattr(self.args, "begin_pretrain", False) and show_progress:
            try:
                from tqdm import tqdm  # type: ignore
            except ImportError:  # pragma: no cover - tqdm optional
                tqdm = None
            if tqdm is not None:
                total = None
                try:
                    total = len(self.data_loader)
                except TypeError:
                    total = None
                epoch_id = int(getattr(self.args, "epoch", 0)) + 1
                data_iter = tqdm(
                    self.data_loader,
                    total=total,
                    desc=f"Epoch {epoch_id} valid",
                    leave=False,
                )
        for batch in data_iter:
            head, relation, tail, label = batch
            head = head.to(self.args.device)
            relation = relation.to(self.args.device)
            tail = tail.to(self.args.device)
            label = label.to(self.args.device)
            num += len(head)
            pred = model.predict(head, relation)
            batch_size_range = torch.arange(pred.size()[0], device=self.args.device)
            target_pred = pred[batch_size_range, tail]
            pred = torch.where(label.bool(), -torch.ones_like(pred) * 10000000, pred)
            pred[batch_size_range, tail] = target_pred
            """ rank all candidate entities """
            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                batch_size_range, tail]
            ranks = ranks.float()
            results['count'] = torch.numel(ranks) + results.get('count', 0.0)
            results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
            results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
            for k in [0, 2, 9]:
                results[f'hits{k + 1}'] = torch.numel(
                    ranks[ranks <= (k + 1)]
                ) + results.get(f'hits{k + 1}', 0.0)
        total_time = time.time() - start_time
        count = float(results['count'])
        for key, val in list(results.items()):
            if key != 'count':
                results[key] = round(val / count, 4)
        results['duration'] = total_time
        return results


class ForgetDBatching():
    """
    Test processor for forgetting set
    """

    def __init__(self, args, kg) -> None:
        self.args = args
        self.kg = kg
        self.batch_size = args.batch_size
        self.dataset = ForgetTestDataset(args, kg)
        self.data_loader = DataLoader(
            self.dataset,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=self.dataset.collate_fn,
            generator=torch.Generator().manual_seed(int(args.seed)),
            pin_memory=should_pin_memory(self.args.device)
        )
        # Store last MRR value for unlearning metrics calculation
        self.last_mrr = 0.0

    def process_epoch(self, model):
        """Process evaluation epoch and calculate metrics.

        Args:
            model: The model instance

        Returns:
            dict: Dictionary containing evaluation metrics
        """
        model.eval()
        num = 0

        # Initialize results with standard ranking metrics (MRR/MR/Hits).
        results = {
            'count': 0.0,
            'mean_rank': 0.0,
            'mrr': 0.0,
            'hits1': 0.0,
            'hits3': 0.0,
            'hits10': 0.0
        }

        # Initialize separate trackers for positive and negative samples
        pos_results = {'count': 0.0, 'mrr': 0.0, 'mr': 0.0, 'hits1': 0.0, 'hits3': 0.0, 'hits10': 0.0}
        neg_results = {'count': 0.0, 'mrr': 0.0, 'mr': 0.0, 'hits1': 0.0, 'hits3': 0.0, 'hits10': 0.0}

        # Setup progress tracking
        start_time = time.time()
        last_log_time = start_time
        total_batches = len(self.data_loader)
        log_interval = getattr(self.args, 'log_interval', 10)

        timestep = getattr(self.args, "timestep_test", getattr(self.args, "timestep", None))
        prefix = f"[timestep {timestep}] " if timestep is not None else ""
        print(f"{prefix}Starting FORGET evaluation with {total_batches} batches")

        for batch_id, batch in enumerate(self.data_loader):
            head, relation, tail, label = batch
            head = head.to(self.args.device)
            relation = relation.to(self.args.device)
            tail = tail.to(self.args.device)
            label = label.to(self.args.device)

            batch_size = len(head)
            num += batch_size

            # Model prediction (use wrapper if available to support SDKU biases)
            if hasattr(model, "predict"):
                pred = model.predict(head, relation)
            else:
                pred = model.kge_model.predict(head, relation)
            batch_size_range = torch.arange(pred.size()[0], device=self.args.device)
            target_pred = pred[batch_size_range, tail]
            pred = torch.where(label.bool(), -torch.ones_like(pred) * 10000000, pred)
            pred[batch_size_range, tail] = target_pred

            # Rank of the true tail in a filtered setting.
            # Computing ranks via argsort is very slow on MPS; counting scores greater than the target is O(n)
            # and matches the standard definition rank = 1 + #{e: score(e) > score(true)} (ties are rare).
            ranks = 1.0 + (pred > target_pred.unsqueeze(1)).sum(dim=1).float()

            # Track additional gold tails (other correct answers) if they exist
            gold_mask = label.bool()
            other_gold_mask = gold_mask.clone()
            other_gold_mask[batch_size_range, tail] = False
            other_gold_rows = torch.where(other_gold_mask)[0]

            # Accumulate results for all samples (for traditional metrics)
            results['count'] += batch_size
            results['mean_rank'] += torch.sum(ranks).item()
            results['mrr'] += torch.sum(1.0 / ranks).item()

            for k in [0, 2, 9]:
                results[f'hits{k + 1}'] += torch.sum(ranks <= (k + 1)).item()

            # Accumulate results for the target triples (one per sample)
            pos_results['count'] += batch_size
            pos_results['mrr'] += torch.sum(1.0 / ranks).item()
            pos_results['mr'] += torch.sum(ranks).item()
            for k in [0, 2, 9]:
                pos_results[f'hits{k + 1}'] += torch.sum(ranks <= (k + 1)).item()

            # Accumulate results for any additional gold tails (rare multi-answer cases)
            extra_count = other_gold_rows.numel()
            if extra_count > 0:
                extra_ranks = ranks[other_gold_rows]
                neg_results['count'] += extra_count
                neg_results['mrr'] += torch.sum(1.0 / extra_ranks).item()
                neg_results['mr'] += torch.sum(extra_ranks).item()
                for k in [0, 2, 9]:
                    neg_results[f'hits{k + 1}'] += torch.sum(extra_ranks <= (k + 1)).item()

            # Update progress bar
            current_time = time.time()
            elapsed = current_time - start_time
            progress = (batch_id + 1) / total_batches
            eta = elapsed / progress - elapsed if progress > 0 else 0

            # Create progress bar
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)

            # Calculate speeds and percentages
            percent = progress * 100
            batches_per_sec = (batch_id + 1) / elapsed if elapsed > 0 else 0

            # Print progress bar with ETA
            print(
                f"\rBatch: {batch_id + 1}/{total_batches} [{bar}] {percent:.1f}% ETA: {eta:.1f}s, Speed: {batches_per_sec:.1f} batches/s",
                end="")
        print()

        # SAFETY CHECK: Ensure we processed some data
        if results['count'] == 0:
            print("WARNING: No samples were processed during evaluation!")
            return results

        # Calculate final metrics for traditional metrics
        count = float(results['count'])
        for key, val in results.items():
            if key != 'count':
                results[key] = round(val / count, 4)

        # Calculate final metrics for positive samples
        pos_count = float(pos_results['count'])
        if pos_count > 0:
            for key, val in pos_results.items():
                if key != 'count':
                    pos_results[key] = round(val / pos_count, 4)

        # Calculate final metrics for negative samples
        neg_count = float(neg_results['count'])
        if neg_count > 0:
            for key, val in neg_results.items():
                if key != 'count':
                    neg_results[key] = round(val / neg_count, 4)

        # Calculate Mf, Mr, MAvg, and MF1
        mistake_rate = 1.0 - pos_results.get('hits1', 0.0)
        forget_rate = 1.0 - neg_results.get('hits1', 0.0)
        mavg = (pos_results.get('hits1', 0.0) + neg_results.get('hits1', 0.0)) / 2.0

        # Calculate precision and recall for F1
        recall = pos_results.get('hits1', 0.0)
        true_positives = pos_count * recall
        false_positives = neg_count * forget_rate
        precision = true_positives / (true_positives + false_positives) if (
                                                                                   true_positives + false_positives) > 0.0 else 0.0

        mf1 = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) > 0.0 else 0.0

        # Store MRR for unlearning metrics calculation
        self.last_mrr = results['mrr']

        # Add additional metrics to results
        # Keep legacy keys mf/mr for backward compatibility, but do not overload mean rank.
        results['mf'] = round(forget_rate, 4)
        results['mr'] = round(mistake_rate, 4)
        results['forget_rate'] = results['mf']
        results['mistake_rate'] = results['mr']
        results['mavg'] = round(mavg, 4)
        results['mf1'] = round(mf1, 4)

        # Print summary
        total_time = time.time() - start_time
        samples_per_sec = num / total_time if total_time > 0 else 0
        print(f"{prefix}FORGET evaluation completed in {total_time:.2f}s | "
              f"Processed {num} samples ({samples_per_sec:.1f} samples/s)")
        print(f"Standard metrics - MRR: {results['mrr']:.4f} | "
              f"Hits@1: {results['hits1']:.4f} | "
              f"Hits@3: {results['hits3']:.4f} | "
              f"Hits@10: {results['hits10']:.4f} | "
              f"Mean Rank: {results['mean_rank']:.1f}")
        print(
            f"Aux metrics (NOT paper Mf/Mr/MAvg/MF1) - "
            f"Mf: {results['mf']:.4f} | Mr: {results['mr']:.4f} | "
            f"MAvg: {results['mavg']:.4f} | MF1: {results['mf1']:.4f}"
        )

        # Add evaluation duration
        results['duration'] = total_time

        # Add pos/neg specific metrics to results for potential further analysis
        results['pos_metrics'] = pos_results
        results['neg_metrics'] = neg_results

        return results

    # Getter for last MRR value (for unlearning metrics)
    def get_mrr_f(self):
        """Get the last computed MRR value for the forget dataset."""
        return self.last_mrr


class RetainDBatching():
    """
    Test processor for reserve set
    """

    def __init__(self, args, kg) -> None:
        self.args = args
        self.kg = kg
        self.batch_size = args.batch_size
        self.dataset = RetainTestDataset(args, kg)
        self.data_loader = DataLoader(
            self.dataset,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=self.dataset.collate_fn,
            generator=torch.Generator().manual_seed(int(args.seed)),
            pin_memory=should_pin_memory(self.args.device)
        )
        # Store last MRR value for unlearning metrics calculation
        self.last_mrr = 0.0

    def process_epoch(self, model):
        """Process evaluation epoch and calculate metrics for unlearning.

        Args:
            model: The model instance

        Returns:
            dict: Dictionary containing evaluation metrics
        """
        model.eval()
        num = 0

        # CRITICAL FIX: Initialize results with all required keys
        results = {
            'count': 0.0,
            'mean_rank': 0.0,
            'mrr': 0.0,
            'hits1': 0.0,
            'hits3': 0.0,
            'hits10': 0.0
        }

        # Separate tracking for positive (retain) and negative (forget) samples
        pos_results = {'count': 0.0, 'mrr': 0.0, 'mean_rank': 0.0, 'hits1': 0.0, 'hits3': 0.0, 'hits10': 0.0}
        neg_results = {'count': 0.0, 'mrr': 0.0, 'mean_rank': 0.0, 'hits1': 0.0, 'hits3': 0.0, 'hits10': 0.0}

        # Setup progress tracking
        start_time = time.time()
        total_batches = len(self.data_loader)

        timestep = getattr(self.args, "timestep_test", getattr(self.args, "timestep", None))
        prefix = f"[timestep {timestep}] " if timestep is not None else ""
        print(f"{prefix}Starting RETAIN evaluation with {total_batches} batches")
        full_len = getattr(self.dataset, "_full_test_len", None)
        sampled_len = getattr(self.dataset, "_sampled_test_len", None)
        if isinstance(full_len, int) and isinstance(sampled_len, int) and full_len > 0 and sampled_len > 0 and sampled_len < full_len:
            frac = sampled_len / float(full_len)
            print(
                f"{prefix}RETAIN eval sampling enabled: {sampled_len}/{full_len} triples ({frac:.1%}); "
                f"set `retain_eval_sample_frac` or `retain_eval_sample_size` to control."
            )

        for batch_id, batch in enumerate(self.data_loader):
            head, relation, tail, label = batch
            head = head.to(self.args.device)
            relation = relation.to(self.args.device)
            tail = tail.to(self.args.device)
            label = label.to(self.args.device)
            batch_size = len(head)
            num += batch_size

            # Model prediction (use wrapper if available to support SDKU biases)
            if hasattr(model, "predict"):
                pred = model.predict(head, relation)
            else:
                pred = model.kge_model.predict(head, relation)
            batch_size_range = torch.arange(pred.size()[0], device=self.args.device)
            target_pred = pred[batch_size_range, tail]

            # Mask out positive samples during ranking (filtered setting)
            pred = torch.where(label.bool(), -torch.ones_like(pred) * 10000000, pred)
            pred[batch_size_range, tail] = target_pred

            # Rank of the true tail in a filtered setting.
            # Avoid argsort (very slow on MPS) and compute rank by counting how many candidates beat the target.
            ranks = 1.0 + (pred > target_pred.unsqueeze(1)).sum(dim=1).float()

            # Identify additional gold tails (if any) besides the evaluated target
            gold_mask = label.bool()
            other_gold_mask = gold_mask.clone()
            other_gold_mask[batch_size_range, tail] = False
            other_gold_rows = torch.where(other_gold_mask)[0]

            # REMOVED DEBUG: print("HERE") and exit()

            # Accumulate results for all samples (overall metrics)
            # FIXED: Now results keys are properly initialized
            results['count'] += batch_size
            results['mean_rank'] += torch.sum(ranks).item()
            results['mrr'] += torch.sum(1.0 / ranks).item()
            for k in [0, 2, 9]:
                results[f'hits{k + 1}'] += torch.sum(ranks <= (k + 1)).item()

            # Accumulate results for the primary retain triples (one per sample)
            pos_results['count'] += batch_size
            pos_results['mrr'] += torch.sum(1.0 / ranks).item()
            pos_results['mean_rank'] += torch.sum(ranks).item()
            for k in [0, 2, 9]:
                pos_results[f'hits{k + 1}'] += torch.sum(ranks <= (k + 1)).item()

            # Accumulate results for any additional gold tails (rare multi-answer cases)
            extra_count = other_gold_rows.numel()
            if extra_count > 0:
                extra_ranks = ranks[other_gold_rows]
                neg_results['count'] += extra_count
                neg_results['mrr'] += torch.sum(1.0 / extra_ranks).item()
                neg_results['mean_rank'] += torch.sum(extra_ranks).item()
                for k in [0, 2, 9]:
                    neg_results[f'hits{k + 1}'] += torch.sum(extra_ranks <= (k + 1)).item()

            # Progress tracking
            if (batch_id + 1) % 10 == 0 or batch_id + 1 == total_batches:
                current_time = time.time()
                elapsed = current_time - start_time
                progress = (batch_id + 1) / total_batches
                eta = elapsed / progress - elapsed if progress > 0 else 0

                bar_length = 30
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)

                percent = progress * 100
                batches_per_sec = (batch_id + 1) / elapsed if elapsed > 0 else 0

                print(f"\rBatch: {batch_id + 1}/{total_batches} [{bar}] {percent:.1f}% "
                      f"ETA: {eta:.1f}s, Speed: {batches_per_sec:.1f} batches/s", end="")

        print()

        # SAFETY CHECK: Ensure we processed some data
        if results['count'] == 0:
            print("WARNING: No samples were processed during RETAIN evaluation!")
            return results

        # Calculate final metrics for overall results
        count = float(results['count'])
        for key, val in results.items():
            if key != 'count':
                results[key] = round(val / count, 4)

        # Calculate final metrics for positive samples (retain set)
        pos_count = float(pos_results['count'])
        if pos_count > 0:
            for key, val in pos_results.items():
                if key != 'count':
                    pos_results[key] = round(val / pos_count, 4)

        # Calculate final metrics for negative samples (forget set)
        neg_count = float(neg_results['count'])
        if neg_count > 0:
            for key, val in neg_results.items():
                if key != 'count':
                    neg_results[key] = round(val / neg_count, 4)

        # Calculate unlearning-specific metrics
        # Mistake Rate (Mr): How often the model fails on retain set
        mistake_rate = 1.0 - pos_results.get('hits1', 0.0)

        # Forget Rate (Mf): How often the model fails on forget set (good for unlearning)
        forget_rate = 1.0 - neg_results.get('hits1', 0.0)

        # Average performance
        mavg = (pos_results.get('hits1', 0.0) + neg_results.get('hits1', 0.0)) / 2.0

        # Calculate F1 score for unlearning
        recall = pos_results.get('hits1', 0.0)  # Retain performance
        true_positives = pos_count * recall
        false_positives = neg_count * (1.0 - forget_rate)  # Wrong predictions on forget set
        precision = true_positives / (true_positives + false_positives) if (
                                                                                   true_positives + false_positives) > 0.0 else 0.0
        mf1 = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) > 0.0 else 0.0

        # Store additional unlearning metrics
        self.last_mrr = results['mrr']
        results['mistake_rate'] = round(mistake_rate, 4)
        results['forget_rate'] = round(forget_rate, 4)
        results['mavg'] = round(mavg, 4)
        results['mf1'] = round(mf1, 4)

        # Print comprehensive summary
        total_time = time.time() - start_time
        samples_per_sec = num / total_time if total_time > 0 else 0

        print(f"{prefix}RETAIN evaluation completed in {total_time:.2f}s | "
              f"Processed {num} samples ({samples_per_sec:.1f} samples/s)")

        print(f"Overall metrics - MRR: {results['mrr']:.4f} | "
              f"Hits@1: {results['hits1']:.4f} | "
              f"Hits@3: {results['hits3']:.4f} | "
              f"Hits@10: {results['hits10']:.4f} | "
              f"Mean Rank: {results['mean_rank']:.1f}")

        print(f"Retain set (positive) - MRR: {pos_results['mrr']:.4f} | "
              f"Hits@1: {pos_results['hits1']:.4f} | "
              f"Hits@10: {pos_results['hits10']:.4f}")

        print(f"Forget set (negative) - MRR: {neg_results['mrr']:.4f} | "
              f"Hits@1: {neg_results['hits1']:.4f} | "
              f"Hits@10: {neg_results['hits10']:.4f}")

        print(
            f"Aux metrics (NOT paper Mf/Mr/MAvg/MF1) - "
            f"Mistake Rate: {results['mistake_rate']:.4f} | "
            f"Forget Rate: {results['forget_rate']:.4f} | "
            f"MAvg: {results['mavg']:.4f} | MF1: {results['mf1']:.4f}"
        )

        # Add evaluation duration and detailed results for analysis
        results['duration'] = total_time
        results['retain_metrics'] = pos_results
        results['forget_metrics'] = neg_results

        return results
    # Getter for last MRR value (for unlearning metrics)
    def get_mrr_r(self):
        """Get the last computed MRR value for the reserve dataset."""
        return self.last_mrr
