import json
from collections import defaultdict, Counter
import os
import pickle
import random


class KGBaseTrainingData:
    def __init__(self, args) -> None:
        self.args = args
        self.ent_num = 0
        self.rel_num = 0
        self.entity2id = {}
        self.id2entity = {}
        self.relation2id = {}
        self.id2relation = {}

        # Optional: store entity types; if not provided, default to "Entity"
        self.entity2type = {}  # id:type. For instance, 0: 'Entity'
        self.relation2type = {}  # relation (Domain, Range). For instance: _hypernym': ('Entity', 'Entity')
        self.schema_patterns = {}  # Domain_relation_Range. For instance Entity__hypernym_Entity': ('Entity', '_hypernym', 'Entity')
        self.relation_to_types = {}

        self.hr2t = {}
        self.tr2h = {}
        self.train_data = []

        # Initialize test sets for unlearning
        self.retain_test_set = []
        self.forget_test_set = []

        # First try to load entity and relation mappings from files to ensure consistency
        self.try_load_mappings_from_files()

        # Then load schema patterns and data
        self.load_schema()

        # CRITICAL FIX: Only load training data if we're not loading from pickle
        # This will be controlled by the child class
        if not hasattr(self, '_skip_data_loading'):
            self.load_data()

    def try_load_mappings_from_files(self):
        """
        Try to load entity and relation mappings from files if they exist.
        This ensures consistency with mappings used by other components.
        """
        train_data_dir = os.path.join(self.args.data_path, self.args.data_name)
        entity_mappings_path = os.path.join(train_data_dir, "entity_mappings.json")
        relation_mappings_path = os.path.join(train_data_dir, "relation_mappings.json")

        # Load entity mappings
        if os.path.exists(entity_mappings_path):
            try:
                with open(entity_mappings_path, 'r') as f:
                    mappings = json.load(f)
                    if 'entity_to_id' in mappings and 'id_to_entity' in mappings:
                        self.entity2id = mappings['entity_to_id']
                        # Convert string keys to integers for id2entity
                        self.id2entity = {int(k): v for k, v in mappings['id_to_entity'].items()}
                        # Update entity count
                        self.ent_num = max([int(k) for k in mappings['id_to_entity'].keys()]) + 1
                        print(f"Loaded entity mappings from file: {len(self.entity2id)} entities")
            except Exception as e:
                print(f"Error loading entity mappings from file: {e}")
                # Reset mappings in case of partial loading
                self.entity2id = {}
                self.id2entity = {}
                self.ent_num = 0

        # Load relation mappings
        if os.path.exists(relation_mappings_path):
            try:
                with open(relation_mappings_path, 'r') as f:
                    mappings = json.load(f)

                # Initialize dictionaries if they don't exist
                if not hasattr(self, 'relation2id'):
                    self.relation2id = {}
                if not hasattr(self, 'id2relation'):
                    self.id2relation = {}

                # Extract relation schema if needed
                if len(self.relation_to_types) == 0:
                    if 'relation_to_types' in mappings:
                        self.relation_to_types.update(mappings['relation_to_types'])
                    elif 'relation_schema' in mappings:
                        for relation, schema_info in mappings['relation_schema'].items():
                            if 'domain' in schema_info and 'range' in schema_info:
                                domain = schema_info['domain']
                                range_ = schema_info['range']
                                self.relation_to_types[relation] = (domain, range_)

                # Load relation to ID mappings
                if 'relation_to_id' in mappings:
                    self.relation2id.update(mappings['relation_to_id'])
                    print(f"Loaded relation2id mapping: {len(self.relation2id)} relations")

                # Load ID to relation mappings with integer keys
                if 'id_to_relation' in mappings:
                    # Convert string keys back to integers
                    id_rel_dict = {int(k): v for k, v in mappings['id_to_relation'].items()}
                    self.id2relation.update(id_rel_dict)
                    print(f"Loaded id2relation mapping: {len(self.id2relation)} relations")

                # Update relation count if needed
                if hasattr(self, 'rel_num'):
                    if self.id2relation:
                        used_ids = list(self.id2relation.keys())
                        self.rel_num = max(used_ids) + 1
                    elif self.relation2id:
                        used_ids = list(self.relation2id.values())
                        self.rel_num = max(used_ids) + 1

            except Exception as e:
                print(f"Error loading relation schema: {e}")

    def fact2id(self, h, r, t):
        """
        Convert string triple to ID triple, adding any missing entities/relations to mappings.
        This ensures we can handle triples even if mappings were pre-loaded.
        """
        # Check and add head entity if missing
        if h not in self.entity2id:
            self.entity2id[h] = self.ent_num
            self.id2entity[self.ent_num] = h
            self.ent_num += 1

        # Check and add relation if missing
        if r not in self.relation2id:
            self.relation2id[r] = self.rel_num
            self.id2relation[self.rel_num] = r
            self.rel_num += 1

        # Check and add tail entity if missing
        if t not in self.entity2id:
            self.entity2id[t] = self.ent_num
            self.id2entity[self.ent_num] = t
            self.ent_num += 1

        h_id = self.entity2id[h]
        r_id = self.relation2id[r]
        t_id = self.entity2id[t]

        return (h_id, r_id, t_id)

    def get_enty_type(self, entity_id):
        return self.entity2type.get(entity_id, "Entity")

    def get_inverse_relation_id(self, relation_id: int) -> int:
        """
        Return the ID of the inverse relation.
        Assumes inverse relations are stored using even/odd pairing
        where the forward relation is even and the inverse is odd.
        """
        if relation_id < 0:
            raise ValueError("Relation ID must be non-negative")
        if relation_id % 2 == 0:
            inverse_id = relation_id + 1
        else:
            inverse_id = relation_id - 1
        if inverse_id >= self.rel_num or inverse_id < 0:
            raise IndexError(
                f"Inverse relation id {inverse_id} out of range for relation {relation_id} "
                f"(rel_num={self.rel_num})"
            )
        return inverse_id

    def load_data(self):
        """
        Load training triples from file and assign entity types based on the schema.
        Automatically creates inverse relation triples with r+1 IDs.
        """
        print("Loading training data from triples.txt...")
        train_data_dir = os.path.join(self.args.data_path, self.args.data_name)
        train_data_path = os.path.join(train_data_dir, "triples.txt")

        # Reset data structures to ensure clean state
        self.train_data = []
        # CRITICAL FIX: Use defaultdict to avoid KeyError
        from collections import defaultdict
        self.hr2t = defaultdict(set)
        self.tr2h = defaultdict(set)
        all_relations = set()

        # Process file in a single pass
        with open(train_data_path, "r", encoding="utf-8") as rf:
            for line in rf:
                h, r, t = line.strip().split()
                all_relations.add(r)

                # Map entities and relations to IDs
                h_id, r_id, t_id = self.fact2id(h, r, t)

                # Assign entity types if not already assigned
                if h_id not in self.entity2type and r in self.relation_to_types:
                    self.entity2type[h_id] = self.relation_to_types[r][0]

                if t_id not in self.entity2type and r in self.relation_to_types:
                    self.entity2type[t_id] = self.relation_to_types[r][1]

                # Create inverse relation ID
                inverse_r_id = r_id + 1

                # Add original triple to train_data
                self.train_data.append((h_id, r_id, t_id))

                # Add inverse triple to train_data
                self.train_data.append((t_id, inverse_r_id, h_id))

                # FIXED: Now these will work because defaultdict creates sets automatically
                # Original triple
                self.hr2t[(h_id, r_id)].add(t_id)
                self.tr2h[(t_id, r_id)].add(h_id)

                # Inverse triple
                self.hr2t[(t_id, inverse_r_id)].add(h_id)
                self.tr2h[(h_id, inverse_r_id)].add(t_id)

        print(f"Loaded {len(self.train_data)} training triples (including inverses)")
        print(f"Built hr2t mapping with {len(self.hr2t)} entries")
        print(f"Built tr2h mapping with {len(self.tr2h)} entries")

        # Ensure relation count covers inverse relations added during loading
        if self.train_data:
            max_relation_id = max(r for _, r, _ in self.train_data)
            if max_relation_id + 1 > self.rel_num:
                self.rel_num = max_relation_id + 1
    def create_unlearning_splits(self):
        """
        CORRECTED: For unlearning evaluation, we don't create train/test splits.
        Instead, we use ALL training data and distinguish between:
        - Retain set: All triples EXCEPT forgotten ones (changes per timestep)
        - Forget set: Triples that should be forgotten (from timestep files)

        The 'test' sets are actually the entire retain/forget sets, not separate splits.
        """
        print("Setting up unlearning evaluation - no data splitting needed...")

        # For unlearning, the 'test' sets are:
        # - retain_test_set: Will be ALL training data minus forgotten triples (computed per timestep)
        # - forget_test_set: Will be the forgotten triples from timestep files

        # We don't create splits here - this is handled in load_timesteps() per timestep
        # The retain_test_set will be computed as: all_training_data - forgotten_triples_up_to_timestep_i

        print("Unlearning evaluation will use:")
        print(f"  - Full training data: {len(self.train_data)} triples")
        print(f"  - Retain test: computed per timestep as (all_data - forgotten_data)")
        print(f"  - Forget test: loaded from timestep files")

        # Initialize empty - will be populated per timestep
        self.retain_test_set = []
        self.forget_test_set = []

    def load_test_sets_from_files(self):
        """
        Load test sets from pre-created files.
        """
        print("Loading test sets from pre-created files...")

        retain_test_path = os.path.join(self.args.data_path, self.args.data_name, "retain_test.txt")
        forget_test_path = os.path.join(self.args.data_path, self.args.data_name, "forget_test.txt")

        # Load retain test set
        self.retain_test_set = []
        if os.path.exists(retain_test_path):
            with open(retain_test_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        h, r, t = parts[:3]
                        try:
                            h_id, r_id, t_id = self.fact2id(h, r, t)
                            self.retain_test_set.append((h_id, r_id, t_id))
                        except KeyError as e:
                            print(f"Warning: Skipping unknown entity/relation in retain test: {e}")
                            continue

        # Load forget test set
        self.forget_test_set = []
        if os.path.exists(forget_test_path):
            with open(forget_test_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        h, r, t = parts[:3]
                        try:
                            h_id, r_id, t_id = self.fact2id(h, r, t)
                            self.forget_test_set.append((h_id, r_id, t_id))
                        except KeyError as e:
                            print(f"Warning: Skipping unknown entity/relation in forget test: {e}")
                            continue

        print(f"Loaded test sets from files:")
        print(f"  Retain Test: {len(self.retain_test_set)}")
        print(f"  Forget Test: {len(self.forget_test_set)}")

    def create_splits_from_training_data(self):
        """
        Create train/test splits from the original training data.
        """
        print("Creating splits from training data...")

        # Set seed for reproducible splits
        random.seed(getattr(self.args, 'seed', 42))

        # Convert to list and shuffle
        all_triples = list(self.train_data)
        random.shuffle(all_triples)

        # Create splits (adjust percentages as needed)
        total_size = len(all_triples)

        # Conservative split to ensure we have proper test sets
        train_size = int(0.7 * total_size)  # 70% for training
        retain_test_size = int(0.15 * total_size)  # 15% for retain test
        forget_test_size = int(0.15 * total_size)  # 15% for forget test

        # Ensure we don't exceed total size
        if train_size + retain_test_size + forget_test_size > total_size:
            forget_test_size = total_size - train_size - retain_test_size

        # Create the splits
        actual_train_data = all_triples[:train_size]
        self.retain_test_set = all_triples[train_size:train_size + retain_test_size]
        self.forget_test_set = all_triples[
                               train_size + retain_test_size:train_size + retain_test_size + forget_test_size]

        # Update train_data to be the actual training set (not the test sets)
        self.train_data = actual_train_data

        print(f"Data splits created:")
        print(f"  Training: {len(actual_train_data)} ({len(actual_train_data) / total_size:.1%})")
        print(f"  Retain Test: {len(self.retain_test_set)} ({len(self.retain_test_set) / total_size:.1%})")
        print(f"  Forget Test: {len(self.forget_test_set)} ({len(self.forget_test_set) / total_size:.1%})")

        # Optionally save the splits for future use
        self.save_test_splits()

    def save_test_splits(self):
        """
        Save the created test splits to files for future consistency.
        """
        try:
            data_dir = os.path.join(self.args.data_path, self.args.data_name)

            # Save retain test set
            retain_test_path = os.path.join(data_dir, "retain_test.txt")
            with open(retain_test_path, 'w', encoding='utf-8') as f:
                for h_id, r_id, t_id in self.retain_test_set:
                    h_name = self.id2entity[h_id]
                    r_name = self.id2relation[r_id]
                    t_name = self.id2entity[t_id]
                    f.write(f"{h_name}\t{r_name}\t{t_name}\n")

            # Save forget test set
            forget_test_path = os.path.join(data_dir, "forget_test.txt")
            with open(forget_test_path, 'w', encoding='utf-8') as f:
                for h_id, r_id, t_id in self.forget_test_set:
                    h_name = self.id2entity[h_id]
                    r_name = self.id2relation[r_id]
                    t_name = self.id2entity[t_id]
                    f.write(f"{h_name}\t{r_name}\t{t_name}\n")

            print(f"Saved test splits to files:")
            print(f"  {retain_test_path}")
            print(f"  {forget_test_path}")

        except Exception as e:
            print(f"Warning: Failed to save test splits: {e}")

    def load_schema(self):
        """
        Load schema definitions from a file in a generic way.
        The file (e.g., schema.txt) should have lines in the format:
            DOMAIN Relation RANGE
        For example:
            Entity _hypernym Entity
            LexicalItem _derivationally_related_form LexicalItem
            ...
        This method builds:
          - self.schema_patterns: a dictionary mapping "DOMAIN_Relation_RANGE" to (DOMAIN, Relation, RANGE)
          - self.relation_to_types: a dictionary mapping each Relation to its (DOMAIN, RANGE)
        """
        train_data_dir = os.path.join(self.args.data_path, self.args.data_name)
        schema_path = os.path.join(train_data_dir, "schema.txt")

        if not os.path.exists(schema_path):
            print(f"Schema file not found: {schema_path}")
            return

        with open(schema_path, "r", encoding="utf-8") as rf:
            lines = [line.strip() for line in rf.readlines() if line.strip()]
            for line in lines:
                parts = line.split()
                if len(parts) != 3:
                    continue  # Skip malformed lines
                domain, rel, range_ = parts
                key = f"{domain}_{rel}_{range_}"
                self.schema_patterns[key] = (domain, rel, range_)
                self.relation_to_types[rel] = (domain, range_)


class KGUnlearningData(KGBaseTrainingData):
    def __init__(self, args) -> None:
        # CRITICAL FIX: Skip automatic data loading in parent class
        self._skip_data_loading = True

        super(KGUnlearningData, self).__init__(args)

        self.timesteps = {i: DatasetTimestep(self.args) for i in range(int(self.args.timesteps_num))}
        self.h2rt = defaultdict(list)  # head -> [(relation, tail), ...]
        self.r2ht = defaultdict(list)  # relation -> [(head, tail), ...]
        self.t2hr = defaultdict(list)  # tail -> [(head, relation), ...]

        # Initialize schema store and type information
        self.schema_store = {}
        self.entity_types = {}  # Use direct dictionary instead of defaultdict to catch missing types
        self.schema_patterns = {}  # Initialize schema patterns dictionary
        self.relation_to_pattern = {}  # Maps relation name directly to its pattern for O(1) lookup
        self.relation_to_types = {}  # Maps relation name to (domain, range) tuple
        self.global_pattern_cache = {}  # Cache to ensure consistent typing across timesteps

        # Load schema if using a schema-based method
        if any(token in args.unlearning_method for token in ("SGKU", "SDKU")):
            schema_path = os.path.join(self.args.data_path, self.args.data_name)
            # Direct load approach - load all pre-generated files first
            self.direct_load_schema_files(schema_path)
            # Initialize schema methods if direct load was incomplete
            if not self.schema_store:
                print("***ERROR: impossible to load schema_store - KnowledgeGraph __init__")
                exit(1)

            # Preload relation domain/range information
            self.relation_domain_range = {}
            # Method 1: Use relation_to_types directly (already loaded from files)
            if hasattr(self, 'relation_to_types'):
                for relation, (domain, range_) in self.relation_to_types.items():
                    self.relation_domain_range[relation] = (domain, range_)
            # Method 2: Extract from schema_patterns as backup
            elif hasattr(self, 'schema_patterns') and self.schema_patterns:
                # Group patterns by relation for quick lookup
                relation_patterns = {}
                for pattern, value in self.schema_patterns.items():
                    if isinstance(pattern, tuple) and len(pattern) == 3:
                        h_type, relation, t_type = pattern
                        if relation not in relation_patterns:
                            relation_patterns[relation] = []
                        relation_patterns[relation].append((h_type, t_type))
                # Extract most common domain/range for each relation
                for relation, type_pairs in relation_patterns.items():
                    if type_pairs:
                        # Use most common domain/range pair
                        counter = Counter(type_pairs)
                        most_common = counter.most_common(1)[0][0]
                        self.relation_domain_range[relation] = most_common
            # Quality assessment if requested
            self.assess_schema_quality(args)

        # CRITICAL FIX: Load timesteps FIRST, which will load training data efficiently from pickle
        self.load_timesteps()

        # ONLY create splits if we don't have proper test sets loaded from pickle
        # CORRECTED: For unlearning, we don't need separate test splits
        # The retain/forget sets are computed per timestep based on forgotten triples
        if not hasattr(self, 'train_data') or not self.train_data:
            self.create_unlearning_splits()

    def direct_load_schema_files(self, data_path):
        """
        Directly load schema information from JSON files to avoid serialization issues.
        Works directly with schema_store.json rather than using pickle files.
        """
        # Always prioritize loading from schema_store.json which contains proper entity sets
        schema_store_path = os.path.join(data_path, "schema_store.json")
        files_loaded = 0
        if os.path.exists(schema_store_path):
            try:
                with open(schema_store_path, 'r') as f:
                    json_schema_store = json.load(f)
                # Convert from JSON-compatible format to internal representation
                self.schema_store = {}
                total_head_entities = 0
                total_tail_entities = 0
                missing_head = 0
                missing_tail = 0

                # Process each pattern in the schema store
                for pattern_str, value in json_schema_store.items():
                    # Parse pattern string back to tuple
                    try:
                        pattern_str = pattern_str.strip('()')
                        parts = pattern_str.split(',')
                        if len(parts) >= 3:
                            domain = parts[0].strip().strip("'\"")
                            relation = parts[1].strip().strip("'\"")
                            range_ = ','.join(parts[2:]).strip().strip("'\"")
                            pattern_key = (domain, relation, range_)
                        else:
                            pattern_key = tuple(pattern_str.split(' ', 2))
                    except Exception as e:
                        print(f"Warning: Could not parse pattern string: {pattern_str}. Error: {str(e)}")
                        exit(1)

                    # Create a new entry for this pattern
                    self.schema_store[pattern_key] = {
                        'importance': value.get('importance', 1.0),
                        'entities': {
                            'head': set(),
                            'tail': set()
                        },
                        'relations': set()
                    }

                    # Process the entity sets
                    if 'entities' in value and isinstance(value['entities'], dict):
                        # Process head entities
                        if 'head' in value['entities']:
                            head_entities = value['entities']['head']
                            if isinstance(head_entities, list) and head_entities:
                                entity_set = set(
                                    int(e) if isinstance(e, str) and e.isdigit() else e for e in head_entities)
                                self.schema_store[pattern_key]['entities']['head'] = entity_set
                                total_head_entities += len(entity_set)
                            else:
                                missing_head += 1
                        else:
                            missing_head += 1

                        # Process tail entities
                        if 'tail' in value['entities']:
                            tail_entities = value['entities']['tail']
                            if isinstance(tail_entities, list) and tail_entities:
                                entity_set = set(
                                    int(e) if isinstance(e, str) and e.isdigit() else e for e in tail_entities)
                                self.schema_store[pattern_key]['entities']['tail'] = entity_set
                                total_tail_entities += len(entity_set)
                            else:
                                missing_tail += 1
                        else:
                            missing_tail += 1

                    # Process relations
                    if 'relations' in value:
                        if isinstance(value['relations'], list):
                            self.schema_store[pattern_key]['relations'] = set(value['relations'])
                        elif isinstance(value['relations'], str):
                            self.schema_store[pattern_key]['relations'] = {value['relations']}

                    # Store relation-to-pattern mapping for each relation in this pattern
                    for rel in self.schema_store[pattern_key]['relations']:
                        self.relation_to_pattern[rel] = pattern_key

                    # Also add the main pattern relation
                    if pattern_key[1] not in self.relation_to_pattern:
                        self.relation_to_pattern[pattern_key[1]] = pattern_key

                # Store domain and range types from patterns
                for pattern, data in self.schema_store.items():
                    domain, relation, range_ = pattern
                    if relation not in self.relation_to_types:
                        self.relation_to_types[relation] = (domain, range_)

                files_loaded += 1
            except Exception as e:
                print(f"Error loading schema store JSON: {e}")

        # Load entity types from entity_types.json
        entity_types_path = os.path.join(data_path, "entity_types.json")
        if os.path.exists(entity_types_path):
            try:
                with open(entity_types_path, 'r') as f:
                    entity_types_dict = json.load(f)

                # Initialize entity_types with empty sets if needed
                if not hasattr(self, 'entity_types'):
                    self.entity_types = {}

                # Map entities by ID or string
                for entity, type_names in entity_types_dict.items():
                    # Convert single string to list for consistent processing
                    if isinstance(type_names, str):
                        type_names = [type_names]

                    # Create a set from the list of types
                    type_set = set(type_names)

                    # Try to match by entity name
                    if entity in self.entity2id:
                        entity_id = self.entity2id[entity]
                        self.entity_types[entity_id] = type_set

                    # Try numeric ID approach
                    try:
                        entity_id = int(entity)
                        if entity_id in self.id2entity:
                            self.entity_types[entity_id] = type_set
                    except:
                        pass

                files_loaded += 1
            except Exception as e:
                print(f"Error loading entity types: {e}")

        # Load relation mappings if needed
        if len(self.relation_to_types) == 0:
            relation_mappings_path = os.path.join(data_path, "relation_mappings.json")
            if os.path.exists(relation_mappings_path):
                try:
                    with open(relation_mappings_path, 'r') as f:
                        mappings = json.load(f)

                    # Extract relation schema
                    if 'relation_to_types' in mappings:
                        self.relation_to_types.update(mappings['relation_to_types'])
                    elif 'relation_schema' in mappings:
                        for relation, schema_info in mappings['relation_schema'].items():
                            if 'domain' in schema_info and 'range' in schema_info:
                                domain = schema_info['domain']
                                range_ = schema_info['range']
                                self.relation_to_types[relation] = (domain, range_)

                    # Load relation to ID mappings
                    if 'relation_to_id' in mappings:
                        self.relation2id = mappings['relation_to_id']

                    # Load ID to relation mappings with integer keys
                    if 'id_to_relation' in mappings:
                        # Convert string keys back to integers
                        self.id2relation = {int(k): v for k, v in mappings['id_to_relation'].items()}

                    files_loaded += 1
                except Exception as e:
                    print(f"Error loading relation schema: {e}")

        # Build schema patterns if needed
        if not self.schema_patterns and self.schema_store:
            for pattern in self.schema_store.keys():
                self.schema_patterns[pattern] = pattern

        # If we have schema_store and relation_to_pattern, no need to use pickles
        if self.schema_store and self.relation_to_pattern:
            return True

    def load_timesteps(self):
        """
        FIXED: Load timestep data efficiently from pickle files.
        Only loads training data from triples.txt if pickle doesn't exist.
        """
        # Initialize all required data structures upfront
        self.relation_to_types = getattr(self, 'relation_to_types', {})
        self.relation_to_pattern = getattr(self, 'relation_to_pattern', {})
        self.global_pattern_cache = getattr(self, 'global_pattern_cache', {})

        # Initialize index structures with defaultdict to avoid key checks
        self.h2rt = getattr(self, 'h2rt', defaultdict(list))
        self.r2ht = getattr(self, 'r2ht', defaultdict(list))
        self.t2hr = getattr(self, 't2hr', defaultdict(list))

        # Determine if we're using schema-based method
        using_schema = any(token in self.args.unlearning_method for token in ("SGKU", "SDKU"))

        kge_suffix = getattr(self.args, "kge", "").lower()
        base_dir = os.path.join(self.args.data_path, self.args.data_name)
        candidate_dirs = []
        if kge_suffix:
            candidate_dirs.append(os.path.join(base_dir, "forget_sets", kge_suffix, "timesteps"))
            candidate_dirs.append(os.path.join(base_dir, f"timesteps_{kge_suffix}"))
        candidate_dirs.append(os.path.join(base_dir, "timesteps"))

        timesteps_dir = None
        for cdir in candidate_dirs:
            if os.path.isdir(cdir):
                timesteps_dir = cdir
                break

        if timesteps_dir is None:
            raise FileNotFoundError(
                f"No timesteps directory found for dataset '{self.args.data_name}' and model '{self.args.kge}'."
            )

        timesteps_parent = os.path.dirname(timesteps_dir)
        timesteps_pickle_name = f"timesteps_{kge_suffix}_fixed.pkl" if kge_suffix else "timesteps_fixed.pkl"
        timesteps_pickle_path = os.path.join(timesteps_parent, timesteps_pickle_name)

        # Try to load from pickle first (fast path)
        if os.path.exists(timesteps_pickle_path):
            try:
                print(f"Loading timesteps from pickle file: {timesteps_pickle_path}")
                with open(timesteps_pickle_path, 'rb') as f:
                    timesteps_data = pickle.load(f)

                if 'timesteps' in timesteps_data and len(timesteps_data['timesteps']) == len(self.timesteps):
                    # CRITICAL FIX: Load training data from pickle instead of file
                    if 'train_data' in timesteps_data:
                        self.train_data = timesteps_data['train_data']
                        print(f"Loaded training data from pickle: {len(self.train_data)} triples")

                    # Load hr2t and tr2h mappings from pickle
                    if 'hr2t' in timesteps_data:
                        self.hr2t = timesteps_data['hr2t']
                    if 'tr2h' in timesteps_data:
                        self.tr2h = timesteps_data['tr2h']

                    # Fast path: load all timestep data from pickle
                    for i, snapshot_data in enumerate(timesteps_data['timesteps']):
                        for key, value in snapshot_data.items():
                            setattr(self.timesteps[i], key, value)

                    # Load cached data structures
                    for key in ['global_pattern_cache', 'h2rt', 'r2ht', 't2hr', 'schema_patterns']:
                        if key in timesteps_data:
                            setattr(self, key, timesteps_data[key])

                    # Load schema-related data
                    for key in ['relation_to_pattern', 'relation_to_types']:
                        if key in timesteps_data:
                            getattr(self, key).update(timesteps_data[key])

                    if self.train_data:
                        max_relation_id = max(r for _, r, _ in self.train_data)
                        if max_relation_id + 1 > self.rel_num:
                            self.rel_num = max_relation_id + 1

                    print("Successfully loaded all data from pickle cache - NO triples.txt loading needed!")
                    return  # Successfully loaded from pickle, exit early
                else:
                    print(f"Snapshot count mismatch in pickle ({len(timesteps_data.get('timesteps', []))}) "
                          f"vs requested ({len(self.timesteps)}). Loading from files...")
            except Exception as e:
                print(f"Error loading timesteps from pickle: {e}")

        # FALLBACK: Only load training data if not loaded from pickle
        print("Pickle not available - loading training data from triples.txt...")
        self.load_data()  # This calls the parent method to load from triples.txt

        # Create test splits if not loaded from pickle
        # CORRECTED: For unlearning, we don't create separate test splits
        # We use the full training data and compute retain/forget per timestep
        if not self.train_data:
            print("No training data found - this shouldn't happen!")
            self.create_unlearning_splits()

        # Initialize accumulators for all timesteps
        all_forgotten_triples = [set() for _ in range(int(self.args.timesteps_num))]
        all_forgotten_patterns = [set() for _ in range(int(self.args.timesteps_num))]

        # Cache for triple to entity ID conversions
        triple_id_cache = {}

        # Process each timestep file
        for i in range(int(self.args.timesteps_num)):
            timestep = self.timesteps[i]

            # Initialize timestep attributes
            timestep.forgotten_triples = []
            timestep.forgotten_entities = []
            timestep.is_heads = []
            timestep.retain_triples = []

            if using_schema:
                timestep.forgotten_patterns = []
                timestep.retain_patterns = []

            # Process timestep file
            unlearning_data_path = os.path.join(
                timesteps_dir, f"{i}.txt"
            )

            print(f"\nProcessing timestep {i} from {unlearning_data_path}")

            # Stream file instead of reading all lines at once
            with open(unlearning_data_path, "r", encoding="utf-8") as rf:
                for line in rf:
                    # Parse line
                    parts = line.strip().split("\t")
                    if len(parts) == 4:
                        head, relation, tail, forget_entity = parts
                    elif len(parts) == 3:
                        head, relation, tail = parts
                        forget_entity = head  # Default to head
                    else:
                        continue  # Skip malformed lines

                    # Update index structures efficiently
                    self.h2rt[head].append((relation, tail))
                    self.r2ht[relation].append((head, tail))
                    self.t2hr[tail].append((head, relation))

                    # Determine forgotten entity position
                    is_head = forget_entity == head

                    # Convert to IDs (with caching)
                    triple_key = (head, relation, tail)
                    if triple_key not in triple_id_cache:
                        triple_id_cache[triple_key] = self.fact2id(head, relation, tail)
                    h_id, r_id, t_id = triple_id_cache[triple_key]

                    triple = (h_id, r_id, t_id)
                    forget_entity_id = self.entity2id[forget_entity]

                    # Store triple information
                    timestep.forgotten_triples.append(triple)
                    timestep.forgotten_entities.append(forget_entity_id)
                    timestep.is_heads.append(is_head)

                    # Process schema patterns if needed
                    if using_schema:
                        pattern = None
                        # Efficient pattern lookup with caching
                        if triple_key in self.global_pattern_cache:
                            pattern = self.global_pattern_cache[triple_key]
                        elif relation in self.relation_to_pattern:
                            pattern = self.relation_to_pattern[relation]
                        elif relation in self.relation_to_types:
                            domain_type, range_type = self.relation_to_types[relation]
                            pattern = (domain_type, relation, range_type)
                            self.global_pattern_cache[triple_key] = pattern

                        timestep.forgotten_patterns.append(pattern)

            # Accumulate forgotten triples efficiently
            # For the current timestep i, include all triples from timesteps 0 to i
            all_forgotten_triples[i] = set(timestep.forgotten_triples)
            if i > 0:
                all_forgotten_triples[i].update(all_forgotten_triples[i - 1])

            # Store accumulated triples
            timestep.total_forg_triples = all_forgotten_triples[i]

            # Handle patterns similarly if using schema
            if using_schema:
                # Start with current timestep's patterns (filtering None values)
                all_forgotten_patterns[i] = {p for p in timestep.forgotten_patterns if p is not None}
                if i > 0:
                    all_forgotten_patterns[i].update(all_forgotten_patterns[i - 1])

                timestep.total_forg_patterns = all_forgotten_patterns[i]

            # CRITICAL FIX: Use accumulated forgotten triples and self.train_data
            accumulated_forgotten_set = all_forgotten_triples[i]  # All forgotten triples up to timestep i

            # CORRECTED: Use self.train_data instead of retain_test_set/forget_test_set
            timestep.retain_triples = [t for t in self.train_data if t not in accumulated_forgotten_set]
            timestep.forget_test_triples = list(accumulated_forgotten_set)

            # Process retained patterns if using schema
            if using_schema:
                timestep.retain_patterns = []
                for triple in timestep.retain_triples:
                    h_id, r_id, t_id = triple
                    r_name = self.get_relation_type(r_id)

                    # Efficient pattern lookup
                    pattern = None
                    if r_name in self.relation_to_pattern:
                        pattern = self.relation_to_pattern[r_name]
                    elif r_name in self.relation_to_types:
                        domain_type, range_type = self.relation_to_types[r_name]
                        pattern = (domain_type, r_name, range_type)

                    timestep.retain_patterns.append(pattern)

            # Store statistics
            timestep.forgotten_num = len(timestep.forgotten_entities)
            timestep.acc_forgotten_num = len(timestep.total_forg_triples)
            timestep.reserve_num = len(timestep.retain_triples)
            timestep.forget_test_num = len(getattr(timestep, 'forget_test_triples', []))

            # Report meaningful statistics
            print(f"Timestep {i}:")
            print(f"  Forgotten entities: {timestep.forgotten_num}")
            print(f"  Accumulated forgotten triples: {timestep.acc_forgotten_num}")
            print(f"  Retain test triples: {timestep.reserve_num}")
            print(f"  Forget test triples: {timestep.forget_test_num}")

        # Ensure relation count accounts for inverse relations (even when loading from cache)
        if self.train_data:
            max_relation_id = max(r for _, r, _ in self.train_data)
            if max_relation_id + 1 > self.rel_num:
                self.rel_num = max_relation_id + 1

        # Save to pickle for future use (including training data and test sets)
        try:
            # Create dictionary of timestep data
            timesteps_data = {
                'train_data': self.train_data,  # CRITICAL FIX: Save training data to pickle
                'hr2t': dict(self.hr2t),  # Save hr2t mappings
                'tr2h': dict(self.tr2h),  # Save tr2h mappings
                'timesteps': [
                    {
                        'forgotten_triples': timestep.forgotten_triples,
                        'forgotten_entities': timestep.forgotten_entities,
                        'is_heads': timestep.is_heads,
                        'retain_triples': timestep.retain_triples,
                        'forget_test_triples': getattr(timestep, 'forget_test_triples', []),
                        'forgotten_num': timestep.forgotten_num,
                        'acc_forgotten_num': timestep.acc_forgotten_num,
                        'reserve_num': timestep.reserve_num,
                        'forget_test_num': getattr(timestep, 'forget_test_num', 0),
                        'total_forg_triples': timestep.total_forg_triples,
                        'class_to_forget': getattr(timestep, 'class_to_forget', [])
                    }
                    for timestep in self.timesteps.values()
                ],
                'global_pattern_cache': self.global_pattern_cache,
                'h2rt': dict(self.h2rt),  # Convert defaultdict to dict for serialization
                'r2ht': dict(self.r2ht),
                't2hr': dict(self.t2hr),
                'relation_to_pattern': self.relation_to_pattern,
                'relation_to_types': self.relation_to_types
            }

            # Add schema-specific data if needed
            if using_schema:
                for i, timestep in self.timesteps.items():
                    timesteps_data['timesteps'][i]['forgotten_patterns'] = timestep.forgotten_patterns
                    timesteps_data['timesteps'][i]['retain_patterns'] = timestep.retain_patterns
                    timesteps_data['timesteps'][i]['total_forg_patterns'] = timestep.total_forg_patterns

                if hasattr(self, 'schema_patterns'):
                    timesteps_data['schema_patterns'] = self.schema_patterns

            # Save to pickle
            with open(timesteps_pickle_path, 'wb') as f:
                pickle.dump(timesteps_data, f)
            print(f"✓ Saved complete data (including training data) to pickle file: {timesteps_pickle_path}")
        except Exception as e:
            print(f"Warning: Failed to save timesteps to pickle: {e}")

        print("\nAll timesteps loaded with proper unlearning evaluation setup and cached for future use")
        print(f"SUMMARY - Training: {len(self.train_data)} triples")
        print("Retain/Forget test sets computed per timestep based on forgotten triples")

    # Add schema-related methods
    def get_entity_type(self, entity_id):
        """Get entity type for the given entity ID"""
        return list(self.entity_types.get(entity_id))[0]  # return the first type

    def get_relation_type(self, relation_id):
        """
        Get relation type for the given relation ID.

        This method ALWAYS returns a usable relation name that can be used
        for schema pattern lookup, with priority given to id2relation mapping.

        Args:
            relation_id: The relation ID or name to look up

        Returns:
            String relation name that can be used for schema pattern lookup
        """
        # First priority: direct lookup in id2relation (most common case)
        if hasattr(self, 'id2relation') and relation_id in self.id2relation:
            return self.id2relation[relation_id]

    def get_pattern_and_importance(self, head_id, relation_id, tail_id):
        """
        Get importance value for a schema pattern directly from the schema store.

        This efficient method retrieves the importance value using the preloaded schema_store
        rather than recalculating it, with multiple lookup strategies for reliability.

        Args:
            head_id: Head entity ID
            relation_id: Relation ID
            tail_id: Tail entity ID

        Returns:
            tuple: (pattern, importance_value)
        """
        # Try pattern lookup methods in order of efficiency
        pattern = None

        # Method 1: Get pattern via the relation (most efficient, O(1) lookup)
        r_name = self.get_relation_type(relation_id)
        if hasattr(self, 'relation_to_pattern') and r_name in self.relation_to_pattern:
            pattern = self.relation_to_pattern[r_name]
        # Method 2: Use the schema pattern helper method
        if pattern is None and hasattr(self, 'check_triple_schema'):
            # Get schema pattern for this triple
            h_type = self.get_entity_type(head_id)
            t_type = self.get_entity_type(tail_id)
            pattern = (h_type, r_name, t_type)

        # Method 3: Construct a pattern directly using types
        if pattern is None:
            # Get types via relation domain/range if available
            if hasattr(self, 'relation_to_types') and r_name in self.relation_to_types:
                domain_type, range_type = self.relation_to_types[r_name]
                pattern = (domain_type, r_name, range_type)
            # Fallback to entity-specific types
            elif hasattr(self, 'entity_types'):
                h_type = self.entity_types.get(head_id, "Entity")
                t_type = self.entity_types.get(tail_id, "Entity")
                h_list = list(h_type)
                t_list = list(t_type)
                pattern = (h_list[0], r_name, t_list[0])
            else:
                # Last resort if no type information available
                print("ERROR Knowledge Graph: no pattern found!")
                exit()
        # Now get the importance value from the schema store
        if pattern and hasattr(self, 'schema_store') and pattern in self.schema_store:
            schema_info = self.schema_store[pattern]
            if 'importance' in schema_info:
                return pattern, schema_info['importance']

        # Default importance value if pattern not found or no importance set
        return pattern, 0.5

    def assess_schema_quality(self, args):
        """
        Simplified schema quality assessment that doesn't require SchemaAdapter.
        """
        print("------------------------")
        if hasattr(args, 'assess_schema') and args.assess_schema:
            print("Schema quality assessment is disabled - enable it by setting assess_schema to True in args.")
            print("Schema information is now directly loaded from pre-computed files.")
            print("Please run CreateSchema.py to generate or update schema files.")

            # Report basic schema statistics
            print("\nSchema Statistics:")
            print(f"- Total patterns: {len(self.schema_patterns)}")
            print(f"- Schema store entries: {len(self.schema_store)}")
            print(f"- Relation-to-pattern mappings: {len(self.relation_to_pattern)}")

            # Report entity types
            entity_type_count = len(self.entity_types)
            entity_type_ratio = entity_type_count / self.ent_num if self.ent_num > 0 else 0
            print(f"- Entity types: {entity_type_count}/{self.ent_num} ({entity_type_ratio:.1%})")

            # Report importance distribution if available
            if hasattr(self, 'schema_store') and self.schema_store:
                importance_values = [info.get('importance', 1.0) for info in self.schema_store.values()
                                     if isinstance(info, dict) and 'importance' in info]
                if importance_values:
                    min_importance = min(importance_values)
                    max_importance = max(importance_values)
                    avg_importance = sum(importance_values) / len(importance_values)
                    print(
                        f"- Pattern importance range: {min_importance:.2f} - {max_importance:.2f}, avg: {avg_importance:.2f}")

        pass


class DatasetTimestep:
    def __init__(self, args) -> None:
        self.args = args
        self.forgotten_triples = []  # Triples to be forgotten in this timestep
        self.forgotten_entities = []  # Entities to be forgotten in this timestep
        self.forgotten_patterns = []  # Schema patterns for forgotten triples
        self.total_forg_triples = []  # All forgotten triples up to this timestep
        self.total_forg_patterns = []  # All forgotten patterns up to this timestep
        self.is_heads = []  # Whether the forgotten entity is the head (True) or tail (False)
        self.retain_triples = []  # Triples to be retained in this timestep
        self.retain_patterns = []  # Schema patterns for retained triples
        self.forgotten_num = 0  # Number of forgotten entities in this timestep
        self.acc_forgotten_num = 0  # Total number of forgotten triples up to this timestep
        self.reserve_num = 0  # Number of retained triples in this timestep
        self.forget_test_num = 0  # Number of forget test triples
        self.class_to_forget = []  # Entities to forget
        self.forget_test_triples = []  # Test triples for forget evaluation
