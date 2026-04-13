from copy import deepcopy
from enum import Enum, auto
from typing import Any

from torch.utils.data import Dataset

from src.utilities.utilities import *


def _resolve_eval_sample_size(*, total: int, sample_size: Any, sample_frac: Any) -> int:
    """Resolve an evaluation subsample size from either an absolute size or a fraction.

    - `sample_size` takes precedence when > 0
    - `sample_frac` is interpreted as a fraction in (0, 1]
    """
    try:
        size = int(sample_size or 0)
    except Exception:
        size = 0
    if size > 0:
        return min(size, total)

    try:
        frac = float(sample_frac or 0.0)
    except Exception:
        frac = 0.0
    if frac <= 0.0:
        return 0
    if frac > 1.0:
        frac = 1.0
    return max(1, min(total, int(round(total * frac))))


class TrainDataset(Dataset):
    def __init__(self, args, kg) -> None:
        super(TrainDataset, self).__init__()
        self.args = args
        self.kg = kg
        self.facts = self.construct_facts()

    def __len__(self):
        return len(self.facts)

    def __getitem__(self, index):
        ele = self.facts[index]
        fact, label = ele['fact'], ele['label']
        """ negative sampling """
        fact, label = self.corrupt(fact)
        fact, label = torch.LongTensor(fact), torch.Tensor(label)
        return fact, label, None, None

    @staticmethod
    def collate_fn(data):
        """ _: (fact, label, None, None) """
        fact = torch.cat([_[0] for _ in data], dim=0)
        label = torch.cat([_[1] for _ in data], dim=0)
        """ return: (h, r, t, label) """
        """ label: 1/-1 """
        return fact[:, 0], fact[:, 1], fact[:, 2], label

    def construct_facts(self):
        facts = []
        for h, r, t in self.kg.train_data:
            inverse_r = self.kg.get_inverse_relation_id(r)
            facts.append({
                'fact': (h, r, t),
                'label': 1})
            facts.append({
                'fact': (t, inverse_r, h),
                'label': 1
            })
        return facts

    def corrupt(self, fact):
        """ generate pos/neg facts from pos facts """
        h, r, t = fact
        prob = 0.4
        """
        """
        neg_h = np.random.randint(0, self.kg.ent_num - 1, self.args.neg_ratio)
        neg_t = np.random.randint(0, self.kg.ent_num - 1, self.args.neg_ratio)
        pos_h = np.ones_like(neg_h) * h
        pos_t = np.ones_like(neg_t) * t
        rand_prob = np.random.rand(self.args.neg_ratio)
        head = np.where(rand_prob > prob, pos_h, neg_h)
        tail = np.where(rand_prob > prob, neg_t, pos_t)
        facts = [(h, r, t)]
        label = [1]
        for nh, nt in zip(head, tail):
            facts.append((nh, r, nt))
            label.append(-1)
        return facts, label


class TestDataset(Dataset):
    def __init__(self, args, kg) -> None:
        super(TestDataset, self).__init__()
        self.args = args
        self.kg = kg
        self.test = self.construct_facts()

    def __len__(self):
        return len(self.test)

    def __getitem__(self, index):
        ele = self.test[index]
        fact, label = torch.LongTensor(ele['fact']), ele['label']
        label = self.get_label(label)
        return fact[0], fact[1], fact[2], label

    @staticmethod
    def collate_fn(data):
        h = torch.stack([_[0] for _ in data], dim=0)
        r = torch.stack([_[1] for _ in data], dim=0)
        t = torch.stack([_[2] for _ in data], dim=0)
        label = torch.stack([_[3] for _ in data], dim=0)
        return h, r, t, label

    def get_label(self, label):
        """ for valid and test, a label is all entities labels: [0, ..., 0, 1, 0, ..., 0]"""
        y = np.zeros([self.kg.ent_num], dtype=np.float32)
        for e2 in label:
            y[e2] = 1.0
        return torch.FloatTensor(y)

    def construct_facts(self):
        test = []
        for h, r, t in self.kg.train_data:
            inverse_r = self.kg.get_inverse_relation_id(r)
            test.append({
                'fact': (h, r, t),
                'label': self.kg.hr2t[(h, r)]
            })
            test.append({
                'fact': (t, inverse_r, h),
                'label': self.kg.hr2t[(t, inverse_r)]
            })
        sample_n = _resolve_eval_sample_size(
            total=len(test),
            sample_size=getattr(self.args, "eval_sample_size", 0),
            sample_frac=getattr(self.args, "eval_sample_frac", 0.0),
        )
        if sample_n > 0 and len(test) > sample_n:
            rng = random.Random(int(getattr(self.args, "seed", 42)) + 30001)
            test = rng.sample(test, sample_n)
        return test


class RetainTestDataset(Dataset):
    def __init__(self, args, kg) -> None:
        super(RetainTestDataset, self).__init__()
        self.args = args
        self.kg = kg
        self.valid, self.test = self.construct_facts()

    def __len__(self):
        use_valid = getattr(self.args, "valid", False) and len(self.valid) > 0
        if use_valid:
            return len(self.valid)
        return len(self.test)

    def __getitem__(self, index):
        use_valid = getattr(self.args, "valid", False) and len(self.valid) > 0
        if use_valid:
            ele = self.valid[index]
        else:
            ele = self.test[index]
        fact, label = torch.LongTensor(ele['fact']), ele['label']
        label = self.get_label(label)
        return fact[0], fact[1], fact[2], label

    @staticmethod
    def collate_fn(data):
        h = torch.stack([_[0] for _ in data], dim=0)
        r = torch.stack([_[1] for _ in data], dim=0)
        t = torch.stack([_[2] for _ in data], dim=0)
        label = torch.stack([_[3] for _ in data], dim=0)
        return h, r, t, label

    def get_label(self, label):
        """For valid and test, a label is all entities labels: [0, ..., 0, 1, 0, ..., 0]"""
        y = np.zeros([self.kg.ent_num], dtype=np.float32)
        for e2 in label:
            y[e2] = 1.0
        return torch.FloatTensor(y)

    def construct_facts(self):
        valid, test = [], []

        for h, r, t in self.kg.timesteps[self.args.timestep_test].retain_triples:
            inverse_r = self.kg.get_inverse_relation_id(r)
            if (h, r) in self.kg.hr2t:
                test.append({
                    'fact': (h, r, t),
                    'label': self.kg.hr2t[(h, r)]
                })
            if (t, inverse_r) in self.kg.hr2t:
                test.append({
                    'fact': (t, inverse_r, h),
                    'label': self.kg.hr2t[(t, inverse_r)]
                })

        # Retain evaluation is the expensive part; allow dedicated sampling for rapid tuning.
        self._full_test_len = len(test)
        sample_n = _resolve_eval_sample_size(
            total=len(test),
            sample_size=getattr(self.args, "retain_eval_sample_size", 0) or getattr(self.args, "eval_sample_size", 0),
            sample_frac=getattr(self.args, "retain_eval_sample_frac", 0.0) or getattr(self.args, "eval_sample_frac", 0.0),
        )
        if sample_n > 0 and len(test) > sample_n:
            rng = random.Random(int(getattr(self.args, "seed", 42)) + 31001)
            test = rng.sample(test, sample_n)
        self._sampled_test_len = len(test)

        return valid, test


class ForgetTestDataset(Dataset):
    def __init__(self, args, kg) -> None:
        super(ForgetTestDataset, self).__init__()
        self.args = args
        self.kg = kg
        self.valid, self.test = self.construct_facts()

    def __len__(self):
        use_valid = getattr(self.args, "valid", False) and len(self.valid) > 0
        if use_valid:
            return len(self.valid)
        return len(self.test)

    def __getitem__(self, index):
        use_valid = getattr(self.args, "valid", False) and len(self.valid) > 0
        if use_valid:
            ele = self.valid[index]
        else:
            ele = self.test[index]
        fact, label = torch.LongTensor(ele['fact']), ele['label']
        label = self.get_label(label)
        return fact[0], fact[1], fact[2], label

    @staticmethod
    def collate_fn(data):
        h = torch.stack([_[0] for _ in data], dim=0)
        r = torch.stack([_[1] for _ in data], dim=0)
        t = torch.stack([_[2] for _ in data], dim=0)
        label = torch.stack([_[3] for _ in data], dim=0)
        return h, r, t, label

    def get_label(self, label):
        """For valid and test, a label is all entities labels: [0, ..., 0, 1, 0, ..., 0]"""
        y = np.zeros([self.kg.ent_num], dtype=np.float32)
        for e2 in label:
            y[e2] = 1.0
        return torch.FloatTensor(y)

    def construct_facts(self):
        """
        FIXED: Construct forget test facts - simplified to avoid double inverses.
        Since train_data already includes inverses, use forget_test_triples as-is.
        """
        valid, test = [], []

        # Only add triples that exist as-is (no inverse generation)
        for h, r, t in self.kg.timesteps[self.args.timestep_test].forget_test_triples:
            if (h, r) in self.kg.hr2t:
                test.append({
                    'fact': (h, r, t),
                    'label': self.kg.hr2t[(h, r)]
                })

        # Process validation triples (same simplified logic)
        for h, r, t in self.kg.timesteps[self.args.timestep_validation].forget_test_triples:
            if (h, r) in self.kg.hr2t:
                valid.append({
                    'fact': (h, r, t),
                    'label': self.kg.hr2t[(h, r)]
                })

        sample_n = _resolve_eval_sample_size(
            total=len(test),
            sample_size=getattr(self.args, "eval_sample_size", 0),
            sample_frac=getattr(self.args, "eval_sample_frac", 0.0),
        )
        if sample_n > 0 and len(test) > sample_n:
            rng = random.Random(int(getattr(self.args, "seed", 42)) + 32001)
            test = rng.sample(test, sample_n)

        return valid, test
class DatasetType(Enum):
    """Enum to specify the type of dataset behavior needed"""
    MAIN = auto()  # Standard schema-guided dataset
    FORGET = auto()  # Triples to forget
    RETAIN = auto()  # Triples to retain
    BOUNDARY = auto()  # Schema boundary triples


class RetrainDataset(Dataset):
    """ Use retention sets for training """

    def __init__(self, args, kg) -> None:
        super(RetrainDataset, self).__init__()
        self.args = args
        self.kg = kg
        self.facts = self.construct_facts()

    def __len__(self):
        return len(self.facts)

    def __getitem__(self, index):
        ele = self.facts[index]
        fact, label = ele['fact'], ele['label']
        """ negative sampling """
        fact, label = self.corrupt(fact)
        fact, label = torch.LongTensor(fact), torch.Tensor(label)
        return fact, label, None, None

    @staticmethod
    def collate_fn(data):
        """ _: (fact, label, None, None) """
        fact = torch.cat([_[0] for _ in data], dim=0)
        label = torch.cat([_[1] for _ in data], dim=0)
        """ return: (h, r, t, label) """
        """ label: 1/-1 """
        return fact[:, 0], fact[:, 1], fact[:, 2], label

    def construct_facts(self):
        facts = []
        for h, r, t in self.kg.timesteps[self.args.timestep].retain_triples:
            inverse_r = self.kg.get_inverse_relation_id(r)
            facts.append({
                'fact': (h, r, t),
                'label': 1})
            facts.append({
                'fact': (t, inverse_r, h),
                'label': 1
            })
        return facts

    def corrupt(self, fact):
        """ generate pos/neg facts from pos facts """
        h, r, t = fact
        prob = 0.5
        """
        random corrupt heads and tails
        1 pos + 10 neg = 11 samples
        """
        neg_h = np.random.randint(0, self.kg.ent_num - 1, self.args.neg_ratio)
        neg_t = np.random.randint(0, self.kg.ent_num - 1, self.args.neg_ratio)
        pos_h = np.ones_like(neg_h) * h
        pos_t = np.ones_like(neg_t) * t
        rand_prob = np.random.rand(self.args.neg_ratio)
        head = np.where(rand_prob > prob, pos_h, neg_h)
        tail = np.where(rand_prob > prob, neg_t, pos_t)
        facts = [(h, r, t)]
        label = [1]
        for nh, nt in zip(head, tail):
            facts.append((nh, r, nt))
            label.append(-1)
        return facts, label


class UnifiedSchemaGuidedDataset(Dataset):
    """
    Unified dataset class that can function as any of the three specialized datasets:
    SchemaGuidedDataset, ForgetTripleDataset, or RetainTripleDataset
    """

    def __init__(self, args, kg, dataset_type=DatasetType.MAIN):
        """
        Initialize the unified dataset with configurable behavior.

        Args:
            args: Arguments with configuration parameters
            kg: Knowledge graph instance
            dataset_type: The type of dataset behavior to use
        """
        super(UnifiedSchemaGuidedDataset, self).__init__()
        self.args = args
        self.kg = kg
        self.dataset_type = dataset_type

        # Initialize caches for all dataset types
        self._pattern_cache = {}
        self._entity_type_cache = {}
        self._relation_type_cache = {}
        self._schema_importance = {}

        # Initialize distillation masks
        self.entity_distill_mask = torch.zeros(self.kg.ent_num)
        self.relation_distill_mask = torch.zeros(self.kg.rel_num)

        # Initialize schema store and validate requirements
        self._initialize_schema_infrastructure()

        # Build facts based on dataset type
        self.facts, self.pos_facts, self.neg_facts = self.construct_facts()

    def _initialize_schema_infrastructure(self):
        """Initialize and validate schema infrastructure"""
        # For MAIN and BOUNDARY types, we need full schema validation
        if self.dataset_type in [DatasetType.MAIN, DatasetType.BOUNDARY]:
            # Check if schema store exists, initialize if needed
            if not hasattr(self.kg, 'schema_store') or self.kg.schema_store is None:
                self.kg.schema_store = {}
                # Try to build schema from relation types if available
                if hasattr(self.kg, 'relation_types') and self.kg.relation_types:
                    for rel_id, rel_type in self.kg.relation_types.items():
                        if rel_id % 2 == 0:  # Skip inverse relations
                            pattern = ("Entity", rel_type, "Entity")
                            self.kg.schema_store[pattern] = {
                                'importance': 1.0,
                                'entities': {'head': set(), 'tail': set()},
                                'relations': {rel_type}
                            }
                # Create default patterns from relation IDs if no types available
                elif hasattr(self.kg, 'rel_num'):
                    for r in range(0, self.kg.rel_num, 2):  # Skip inverse relations
                        rel_name = f"relation_{r}"
                        pattern = ("Entity", rel_name, "Entity")
                        self.kg.schema_store[pattern] = {
                            'importance': 1.0,
                            'entities': {'head': set(), 'tail': set()},
                            'relations': {rel_name}
                        }

            self.schema_store = self.kg.schema_store

            # Initialize entity types
            if hasattr(self.kg, 'entity_types'):
                self.entity_types = self.kg.entity_types
                self._entity_type_cache = self.kg.entity_types

            # Initialize relation types
            if hasattr(self.kg, 'relation_types') and self.kg.relation_types:
                self._relation_type_cache = self.kg.relation_types
            elif hasattr(self.kg, 'get_relation_type'):
                # Pre-compute for all relations (usually much fewer than entities)
                for r in range(self.kg.rel_num):
                    try:
                        self._relation_type_cache[r] = self.kg.get_relation_type(r)
                    except:
                        self._relation_type_cache[r] = f"relation_{r}"

            # Calculate schema pattern importance if needed
            if hasattr(self.kg, 'schema_store') and self.kg.schema_store:
                for pattern, info in self.kg.schema_store.items():
                    if 'importance' in info:
                        self._schema_importance[pattern] = info['importance']
                    else:
                        # Calculate default importance
                        domain, relation, range_ = pattern
                        # More specific patterns are more important
                        specificity = 0.5
                        if domain != "Entity":
                            specificity += 0.25
                        if range_ != "Entity":
                            specificity += 0.25
                        self._schema_importance[pattern] = specificity

    def __len__(self):
        """Get the number of items in the dataset"""
        return len(self.facts)

    def __getitem__(self, index):
        """Get an item from the dataset based on the dataset type"""
        if self.dataset_type == DatasetType.MAIN:
            # SchemaGuidedDataset behavior
            pos_fact = self.pos_facts[index]
            neg_fact = self.neg_facts[index]
            ele = self.facts[index]
            fact, label = ele['fact'], ele['label']

            # Negative sampling
            fact, label = self.corrupt(fact, neg_fact)
            fact, label = torch.LongTensor(fact), torch.Tensor(label)
            return fact, label, pos_fact, neg_fact

        elif self.dataset_type == DatasetType.BOUNDARY:
            # SchemaBoundaryDataset behavior
            ele = self.facts[index]
            fact, label = ele['fact'], ele['label']
            importance = ele.get('importance', 1.0)

            # Schema-based corruption
            fact, label = self.schema_corrupt(fact, importance)
            fact, label = torch.LongTensor(fact), torch.Tensor(label)

            # Return the importance as the third element
            return fact, label, importance

        else:
            # ForgetTripleDataset or RetainTripleDataset behavior
            ele = self.facts[index]
            fact, label = ele['fact'], ele['label']
            fact = torch.LongTensor(fact)
            label = torch.Tensor([label])
            return fact, label

    def construct_facts(self):
        """
        Build facts based on the dataset type.

        Returns:
            facts: List of fact dictionaries
            pos_facts: List of positive facts (for MAIN type only)
            neg_facts: List of negative facts (for MAIN type only)
        """
        if self.dataset_type == DatasetType.MAIN:
            return self._build_main_facts()
        elif self.dataset_type == DatasetType.FORGET:
            return self._build_forget_facts(), None, None
        elif self.dataset_type == DatasetType.RETAIN:
            return self._build_retain_facts(), None, None
        elif self.dataset_type == DatasetType.BOUNDARY:
            return self._build_boundary_facts(), None, None
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

    def _build_boundary_facts(self):
        """
        Build boundary facts based on schema patterns of forgotten triples.

        Returns:
            facts: List of boundary facts with schema-based importance
        """
        facts = []
        forgotten_patterns = set()
        schema_affected_entities = set()
        boundary_triples = []

        # Maximum number of boundary triples to use
        max_boundary_triples = getattr(self.args, 'fb_maxnum', 20000)

        # Step 1: Find schema patterns of forgotten triples
        for i, (h, r, t) in enumerate(self.kg.timesteps[self.args.timestep].forgotten_triples):
            pattern, importance = self.kg.get_pattern_and_importance(h, r, t)
            forgotten_patterns.add(pattern)
            # Also track entities directly involved in forgotten triples
            try:
                if self.kg.timesteps[self.args.timestep].is_heads[i]:
                    schema_affected_entities.add(h)
                else:
                    schema_affected_entities.add(t)
            except IndexError:
                # If is_heads information is not available, assume both are affected
                schema_affected_entities.add(h)
                schema_affected_entities.add(t)

        # Step 2: Find triples with similar schema patterns in reserved set
        for h, r, t in self.kg.timesteps[self.args.timestep].retain_triples:
            # Check if triple shares entities with forgotten triples (classic boundary)
            entity_boundary = h in schema_affected_entities or t in schema_affected_entities

            # Check if triple matches schema pattern of any forgotten triple
            pattern, importance = self.kg.get_pattern_and_importance(h, r, t)
            schema_boundary = pattern in forgotten_patterns
            # Include if it's a boundary triple by either definition
            if entity_boundary or schema_boundary:
                boundary_triples.append((h, r, t, importance))

        # Step 3: Sample if too many boundary triples
        if len(boundary_triples) > max_boundary_triples:
            # Sort by importance and take top portion
            top_by_importance = int(max_boundary_triples * 0.7)
            random_selection = max_boundary_triples - top_by_importance

            # Sort by importance (descending)
            boundary_triples.sort(key=lambda x: x[3], reverse=True)

            # Take top important triples
            selected_triples = boundary_triples[:top_by_importance]

            # Randomly sample from remaining
            if random_selection > 0 and len(boundary_triples) > top_by_importance:
                remaining = boundary_triples[top_by_importance:]
                selected_triples.extend(random.sample(remaining, min(random_selection, len(remaining))))
        else:
            selected_triples = boundary_triples

        # Step 4: Create facts including both directions (for symmetry)
        for h, r, t, importance in selected_triples:
            inverse_r = self.kg.get_inverse_relation_id(r)
            facts.append({
                'fact': (h, r, t),
                'label': 1,
                'importance': importance
            })
            facts.append({
                'fact': (t, inverse_r, h),  # Inverse relation
                'label': 1,
                'importance': importance
            })

        # print(facts)
        return facts

    def _build_main_facts(self):
        """
        Build facts for the main schema-guided dataset.

        Returns:
            facts: List of fact dictionaries
            pos_facts: List of positive facts
            neg_facts: List of negative facts
        """
        facts, pos_facts, neg_facts = [], [], []

        # Get the current timestep of the knowledge graph
        timestep = self.kg.timesteps[self.args.timestep]
        forgotten_triples = timestep.forgotten_triples
        reserve_triples = deepcopy(timestep.retain_triples)
        random.shuffle(reserve_triples)

        # Build a dictionary mapping each entity to its associated triples
        reserve_dict = {}
        forgotten_entities = set(timestep.forgotten_entities)

        for h, r, t in reserve_triples:
            reserve_dict.setdefault(h, []).append((h, r, t))
            reserve_dict.setdefault(t, []).append((h, r, t))

            # Update distillation mask: if one entity is forgotten, mark its neighbor
            if h in forgotten_entities:
                self.entity_distill_mask[t] = 1
            if t in forgotten_entities:
                self.entity_distill_mask[h] = 1

        # Optionally reinforce model with untouched retain triples
        retain_sample_ratio = float(getattr(self.args, 'retain_sample_ratio', 0.0))
        if retain_sample_ratio > 0 and reserve_triples:
            sample_count = int(len(forgotten_triples) * retain_sample_ratio)
            sample_count = max(1, sample_count)
            sample_count = min(sample_count, len(reserve_triples))
            retain_sample = random.sample(reserve_triples, sample_count)
            for h, r, t in retain_sample:
                inverse_r = self.kg.get_inverse_relation_id(r)
                pos_facts.extend([(h, r, t), (t, inverse_r, h)])
                facts.extend([
                    {'fact': (h, r, t), 'label': 1},
                    {'fact': (t, inverse_r, h), 'label': 1}
                ])
                # Create corresponding corrupted negatives to keep alignment
                neg_tail = random.randint(0, self.kg.ent_num - 1)
                if neg_tail == t:
                    neg_tail = (neg_tail + 1) % self.kg.ent_num
                neg_head = random.randint(0, self.kg.ent_num - 1)
                if neg_head == h:
                    neg_head = (neg_head + 1) % self.kg.ent_num
                neg_facts.append((h, r, neg_tail))
                neg_facts.append((neg_head, inverse_r, h))

        # Process each forgotten triple to generate positive and negative facts
        for i, (h, r, t) in enumerate(forgotten_triples):
            try:
                is_head = timestep.is_heads[i]
            except IndexError:
                raise IndexError("Mismatch: 'forgotten_triples' and 'is_heads' lengths do not match.")
            forgotten_entity = h if is_head else t
            # Select replacement entity based on the chosen policy
            random_policy = getattr(self.args, "random_policy", "schema")
            if random_policy == "schema":
                # Get relation name and schema pattern
                r_name = self._get_relation_name(r)
                pattern = self._get_schema_pattern(r, r_name)
                random_label = self._get_compatible_entity(pattern, is_head)
            elif random_policy == "close":
                # Use a connected entity
                random_label = self._get_connected_entity(forgotten_entity, reserve_dict)
            else:
                # Random entity
                random_label = random.randint(0, self.kg.ent_num - 1)
            # Create positive facts by replacing the forgotten entity
            inverse_r = self.kg.get_inverse_relation_id(r)
            if is_head:  # Replace head
                pos_fact1 = (random_label, r, t)  # Forward relation with replaced head
                pos_fact2 = (t, inverse_r, random_label)  # Inverse relation
            else:  # Replace tail
                pos_fact1 = (h, r, random_label)  # Forward relation with replaced tail
                pos_fact2 = (random_label, inverse_r, h)  # Inverse relation

            pos_facts.extend([pos_fact1, pos_fact2])
            facts.extend([
                {'fact': pos_fact1, 'label': 1},
                {'fact': pos_fact2, 'label': 1}
            ])
            # Add the original triple as negative (with its inverse variant)
            neg_facts.append((h, r, t))  # Original triple as negative
            neg_facts.append((t, inverse_r, h))  # Inverse triple as negative
        return facts, pos_facts, neg_facts

    def _build_forget_facts(self):
        """
        Build facts for the forget dataset.

        Returns:
            facts: List of fact dictionaries
        """
        facts = []

        # Get the current timestep of the knowledge graph
        timestep = self.kg.timesteps[self.args.timestep]
        forgotten_triples = timestep.forgotten_triples

        # Create facts from forgotten triples
        for h, r, t in forgotten_triples:
            inverse_r = self.kg.get_inverse_relation_id(r)
            # Add the original triple as a fact with negative label (-1)
            facts.append({'fact': (h, r, t), 'label': -1})

            # Also add the inverse triple with negative label
            facts.append({'fact': (t, inverse_r, h), 'label': -1})

        return facts

    def _build_retain_facts(self):
        """
        Build facts for the retain dataset.

        Returns:
            facts: List of fact dictionaries
        """
        facts = []

        # Get the current timestep of the knowledge graph
        timestep = self.kg.timesteps[self.args.timestep]
        reserve_triples = timestep.retain_triples

        # Optionally filter to only include boundary triples
        if hasattr(self.args, 'use_boundary_only') and self.args.use_boundary_only:
            reserve_triples = self._get_boundary_triples(timestep)

        # Create facts from reserve triples
        for h, r, t in reserve_triples:
            inverse_r = self.kg.get_inverse_relation_id(r)
            # Add the triple as a fact with positive label (1)
            facts.append({'fact': (h, r, t), 'label': 1})

            # Also add the inverse triple with positive label
            facts.append({'fact': (t, inverse_r, h), 'label': 1})

        return facts

    def _get_boundary_triples(self, timestep):
        """Get boundary triples (triples connected to forgotten entities)"""
        boundary_triples = []
        forgotten_entities = set(timestep.forgotten_entities)

        for h, r, t in timestep.retain_triples:
            if h in forgotten_entities or t in forgotten_entities:
                boundary_triples.append((h, r, t))

        return boundary_triples

    def _get_relation_name(self, relation_id):
        """Get relation name for a given relation ID"""
        if hasattr(self.kg, 'id2relation') and relation_id in self.kg.id2relation:
            return self.kg.id2relation[relation_id]
        else:
            raise ValueError("Relation not present in id2relation mapping")

    def _get_schema_pattern(self, relation_id, relation_name):
        """Get schema pattern for a relation"""
        if hasattr(self.kg, 'relation_to_pattern') and relation_name in self.kg.relation_to_pattern:
            return self.kg.relation_to_pattern[relation_name]
        elif hasattr(self.kg, 'get_relation_pattern'):
            return self.kg.get_relation_pattern(relation_id)
        else:
            raise ValueError(f"No schema pattern found for relation {relation_name}")


    def _get_compatible_entity(self, pattern, is_head):
        """
        Get a schema-compatible entity for the given pattern and position.

        Args:
            pattern: The schema pattern tuple (domain, relation, range)
            is_head: Boolean indicating if we need a head (True) or tail (False) entity

        Returns:
            An entity ID compatible with the pattern position
        """
        # Default to random entity (fallback)
        random_entity = random.randint(0, self.kg.ent_num - 1)

        # Validate pattern and schema store
        if not pattern or not hasattr(self.kg, 'schema_store'):
            raise ValueError("No valid pattern or schema_store not available")

        # Check if pattern exists in schema store
        if pattern not in self.kg.schema_store:
            raise ValueError(f"Pattern {pattern} not found in schema store")

        # Get schema info for this pattern
        schema_info = self.kg.schema_store[pattern]

        # Check for entities dictionary
        if 'entities' not in schema_info or not isinstance(schema_info['entities'], dict):
            raise ValueError(f"Pattern {pattern} missing valid entities dictionary")

        # Get position key (head or tail)
        position = 'head' if is_head else 'tail'

        # Check if position exists in entities
        if position not in schema_info['entities']:
            raise ValueError(f"Pattern {pattern} has no '{position}' entities")

        # Get entity set for this position
        entities = schema_info['entities'][position]

        # Handle different entity collection types
        if isinstance(entities, set):
            entities_list = list(entities)
        elif isinstance(entities, list):
            entities_list = entities
        elif isinstance(entities, dict):
            entities_list = list(entities.keys())
        else:
            raise ValueError(f"Entities for {position} has unexpected type: {type(entities)}")

        # If no entities, use fallback
        if not entities_list:
            return random_entity

        # Choose a random compatible entity
        return random.choice(entities_list)

    def _get_connected_entity(self, entity, entity_triples_dict):
        """
        Get entity connected to the given entity.

        Args:
            entity: The entity to find connections for
            entity_triples_dict: Dictionary mapping entities to their triples

        Returns:
            An entity connected to the given entity
        """
        connected_entities = []

        # Find all entities connected to this one
        if entity in entity_triples_dict:
            for h, r, t in entity_triples_dict[entity]:
                if h == entity:
                    connected_entities.append(t)
                else:
                    connected_entities.append(h)

        # Fall back to random if no connected entities
        if not connected_entities:
            return random.randint(0, self.kg.ent_num - 1)

        # Choose random connected entity
        return random.choice(connected_entities)

    def corrupt(self, fact, neg_fact):
        """
        Schema-aware corruption for negative sampling.

        Args:
            fact: The positive fact to corrupt
            neg_fact: The explicit negative fact

        Returns:
            facts: List of corrupted facts
            labels: List of labels (1 for positive, -1 for negative)
        """
        # Extract triple components
        if isinstance(fact, tuple):
            h, r, t = fact
        elif isinstance(fact, dict) and 'fact' in fact:
            h, r, t = fact['fact']

        # Get basic info for negative sampling
        neg_size = self.args.neg_ratio - 1

        # Start with positive fact and add explicit negative
        facts = [(h, r, t)]
        labels = [1]
        facts.append((neg_fact[0], neg_fact[1], neg_fact[2]))
        labels.append(-1)

        # Create remaining negatives
        for _ in range(neg_size):
            if random.random() < 0.5:  # Corrupt head with 50% probability
                neg_head = random.randint(0, self.kg.ent_num - 1)
                facts.append((neg_head, r, t))
            else:  # Corrupt tail
                neg_tail = random.randint(0, self.kg.ent_num - 1)
                facts.append((h, r, neg_tail))
            labels.append(-1)

        return facts, labels

    def schema_corrupt(self, fact, importance=1.0):
        """
        Generate negative samples based on schema-aware corruption.

        Args:
            fact: The positive fact to corrupt
            importance: Schema pattern importance score

        Returns:
            facts: List of corrupted facts
            labels: List of labels (1 for positive, -1 for negative)
        """
        if isinstance(fact, tuple):
            h, r, t = fact
        else:
            h, r, t = fact['fact']

        h_type = self.kg.get_entity_type(h)
        t_type = self.kg.get_entity_type(t)

        # Determine corruption strategy based on importance
        # Higher importance = more sophisticated corruption
        if importance > 0.8:
            # For high importance patterns, use schema-based corruption
            # This makes harder negative examples
            return self._schema_type_corruption(h, r, t, h_type, t_type)
        else:
            # For less important patterns, use standard corruption
            return self._standard_corruption(h, r, t)

    def _schema_type_corruption(self, h, r, t, h_type, t_type):
        """
        Generate negatives by corrupting with entities of the same type.

        Args:
            h: Head entity ID
            r: Relation ID
            t: Tail entity ID
            h_type: Head entity type
            t_type: Tail entity type

        Returns:
            facts: List of corrupted facts
            labels: List of labels (1 for positive, -1 for negative)
        """
        facts = [(h, r, t)]
        labels = [1]

        # Find entities of the same type for more challenging negatives
        same_type_heads = []
        same_type_tails = []

        # Collect entities of the same type (limited sample for efficiency)
        for e in range(min(1000, self.kg.ent_num)):
            if e != h and self.kg.get_entity_type(e) == h_type:
                same_type_heads.append(e)
            if e != t and self.kg.get_entity_type(e) == t_type:
                same_type_tails.append(e)

            # Stop once we have enough
            if len(same_type_heads) >= self.args.neg_ratio and len(same_type_tails) >= self.args.neg_ratio:
                break

        # If we didn't find enough, fall back to random sampling
        if len(same_type_heads) < self.args.neg_ratio // 2 or len(same_type_tails) < self.args.neg_ratio // 2:
            return self._standard_corruption(h, r, t)

        # Create negative samples with 50/50 head/tail corruption
        neg_count = 0
        for _ in range(self.args.neg_ratio):
            if neg_count % 2 == 0 and same_type_heads:
                # Corrupt head
                neg_h = random.choice(same_type_heads)
                facts.append((neg_h, r, t))
            elif same_type_tails:
                # Corrupt tail
                neg_t = random.choice(same_type_tails)
                facts.append((h, r, neg_t))
            else:
                # Fallback
                neg_e = random.randint(0, self.kg.ent_num - 1)
                facts.append((neg_e, r, t))

            labels.append(-1)
            neg_count += 1

        return facts, labels

    def _standard_corruption(self, h, r, t):
        """
        Standard corruption by randomly replacing heads or tails.

        Args:
            h: Head entity ID
            r: Relation ID
            t: Tail entity ID

        Returns:
            facts: List of corrupted facts
            labels: List of labels (1 for positive, -1 for negative)
        """
        import numpy as np

        prob = 0.5
        neg_h = np.random.randint(0, self.kg.ent_num - 1, self.args.neg_ratio)
        neg_t = np.random.randint(0, self.kg.ent_num - 1, self.args.neg_ratio)

        pos_h = np.ones_like(neg_h) * h
        pos_t = np.ones_like(neg_t) * t

        rand_prob = np.random.rand(self.args.neg_ratio)
        head = np.where(rand_prob > prob, pos_h, neg_h)
        tail = np.where(rand_prob > prob, neg_t, pos_t)

        facts = [(h, r, t)]
        label = [1]

        for nh, nt in zip(head, tail):
            facts.append((int(nh), r, int(nt)))
            label.append(-1)

        return facts, label

    @property
    def collate_fn(self):
        """Return the appropriate collate function based on dataset type"""
        if self.dataset_type == DatasetType.MAIN:
            return self.main_collate_fn
        elif self.dataset_type == DatasetType.BOUNDARY:
            return self.boundary_collate_fn
        else:  # FORGET or RETAIN
            return self.forget_retain_collate_fn

    @staticmethod
    def main_collate_fn(data):
        """
        Collate function for MAIN dataset (SchemaGuidedDataset).

        Args:
            data: List of (fact, label, pos_fact, neg_fact) tuples

        Returns:
            h, r, t, label, pos_fact, neg_fact tensors
        """
        # Extract facts and labels
        fact = torch.cat([item[0] for item in data], dim=0)
        label = torch.cat([item[1] for item in data], dim=0)

        # Process pos_fact and neg_fact safely
        try:
            pos_fact = list(item[2] for item in data)
            pos_fact = torch.LongTensor(pos_fact)
            neg_fact = list(item[3] for item in data)
            neg_fact = torch.LongTensor(neg_fact)
            return fact[:, 0], fact[:, 1], fact[:, 2], label, pos_fact, neg_fact
        except (TypeError, ValueError) as e:
            print(f"Warning: Error processing pos_fact/neg_fact: {e}")
            # Create dummy tensors as fallback
            batch_size = fact.size(0)
            dummy_tensor = torch.zeros((batch_size, 3), dtype=torch.long)
            return fact[:, 0], fact[:, 1], fact[:, 2], label, dummy_tensor, dummy_tensor

    @staticmethod
    def boundary_collate_fn(data):
        """
        Collate function for BOUNDARY dataset.

        Args:
            data: List of (fact, label, importance) tuples

        Returns:
            h, r, t, label, importance tensors
        """
        # Extract fact and label
        fact = torch.stack([item[0] for item in data])
        label = torch.cat([item[1] for item in data])

        # Extract importance values (third element from __getitem__)
        importance = torch.tensor([item[2] for item in data], dtype=torch.float)

        return fact[:, 0], fact[:, 1], fact[:, 2], label, importance

    @staticmethod
    def forget_retain_collate_fn(data):
        """
        Collate function for FORGET or RETAIN dataset.

        Args:
            data: List of (fact, label) tuples

        Returns:
            h, r, t, label tensors
        """
        fact = torch.stack([item[0] for item in data])
        label = torch.cat([item[1] for item in data])
        return fact[:, 0], fact[:, 1], fact[:, 2], label
