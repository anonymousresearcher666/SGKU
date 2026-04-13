
# !/usr/bin/env python3
"""
Schema Store Builder for Knowledge Graph Datasets

This script generates schema information from knowledge graph triples.
It supports both Freebase and WordNet datasets with appropriate domain/range extraction.
"""
import os
import json
import collections
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR)

# Default dataset (override via CLI)
DEFAULT_DATASET = "fb15k-237-20"


def extract_domain_range_from_path(relation, dataset_type=None):
    """
    Extract domain and range types based on the dataset type.
    Automatically detects if it's Freebase or WordNet based on relation format.
    """
    # Auto-detect dataset type if not specified
    if dataset_type is None:
        if relation.startswith("/") and "/" in relation[1:]:
            dataset_type = "freebase"
        elif relation.startswith("_") or relation in ["hypernym", "hyponym", "meronym", "holonym"]:
            dataset_type = "wordnet"
        else:
            # Default to generic handling
            dataset_type = "generic"

    if dataset_type == "wordnet":
        return extract_wordnet_domain_range(relation)
    elif dataset_type == "freebase":
        return extract_freebase_domain_range(relation)
    else:
        # Generic fallback
        return "Entity", relation, "Entity"


def extract_wordnet_domain_range(relation):
    """
    Extract domain and range types for WordNet relations.
    WordNet relations typically indicate semantic relationships between synsets.
    """
    # Map of WordNet relations to their typical domain and range types
    WORDNET_RELATION_TYPES = {
        # Lexical relations (word-to-word)
        "_derivationally_related_form": ("Word", "Word"),
        "_antonym": ("Word", "Word"),
        "_pertainym": ("Adjective", "Noun"),

        # Semantic relations (synset-to-synset)
        "_hypernym": ("Synset", "Synset"),  # is-a relation
        "_hyponym": ("Synset", "Synset"),  # inverse of hypernym
        "_member_holonym": ("Synset", "Synset"),  # has member
        "_member_meronym": ("Synset", "Synset"),  # is member of
        "_part_holonym": ("Synset", "Synset"),  # has part
        "_part_meronym": ("Synset", "Synset"),  # is part of
        "_substance_holonym": ("Synset", "Synset"),  # has substance
        "_substance_meronym": ("Synset", "Synset"),  # is substance of
        "_instance_hypernym": ("Instance", "Synset"),  # instance of
        "_instance_hyponym": ("Synset", "Instance"),  # has instance
        "_also_see": ("Synset", "Synset"),
        "_verb_group": ("VerbSynset", "VerbSynset"),
        "_similar_to": ("AdjectiveSynset", "AdjectiveSynset"),
        "_entailment": ("VerbSynset", "VerbSynset"),
        "_cause": ("VerbSynset", "VerbSynset"),

        # Domain relations
        "_member_of_domain_topic": ("Synset", "DomainTopic"),
        "_member_of_domain_region": ("Synset", "DomainRegion"),
        "_member_of_domain_usage": ("Synset", "DomainUsage"),
        "_domain_topic_of": ("DomainTopic", "Synset"),
        "_domain_region_of": ("DomainRegion", "Synset"),
        "_domain_usage_of": ("DomainUsage", "Synset"),

        # Morphosemantic relations
        "_participle_of_verb": ("Adjective", "Verb"),
        "_derived_from": ("Word", "Word"),

        # Additional common WordNet relations
        "_attribute": ("Noun", "Adjective"),
        "_has_part": ("Synset", "Synset"),
        "_member_of_this_domain_topic": ("Synset", "Topic"),
        "_member_of_this_domain_region": ("Synset", "Region"),
        "_member_of_this_domain_usage": ("Synset", "Usage"),
    }

    # Check if relation is in our mapping
    if relation in WORDNET_RELATION_TYPES:
        domain_type, range_type = WORDNET_RELATION_TYPES[relation]
        return domain_type, relation, range_type

    # Handle relations with specific patterns
    if relation.startswith("_") and relation.endswith("_of"):
        # Relations like "_part_of", "_member_of" etc.
        base = relation[1:-3]  # Remove _ prefix and _of suffix
        return "Synset", relation, f"{base.capitalize()}Synset"

    # For unknown relations, try to infer from the relation name
    if "_" in relation:
        # Clean relation name
        clean_relation = relation.strip("_")

        # Check for holonym/meronym patterns
        if "holonym" in clean_relation:
            return "Synset", relation, "Synset"
        elif "meronym" in clean_relation:
            return "Synset", relation, "Synset"

        # Check for hypernym/hyponym patterns
        elif "hypernym" in clean_relation:
            return "Synset", relation, "Synset"
        elif "hyponym" in clean_relation:
            return "Synset", relation, "Synset"

        # Check for domain patterns
        elif "domain" in clean_relation:
            if "topic" in clean_relation:
                return "Synset", relation, "DomainTopic"
            elif "region" in clean_relation:
                return "Synset", relation, "DomainRegion"
            elif "usage" in clean_relation:
                return "Synset", relation, "DomainUsage"
            else:
                return "Synset", relation, "Domain"

        # Check for verb-specific relations
        elif any(verb_rel in clean_relation for verb_rel in ["entail", "cause"]):
            return "VerbSynset", relation, "VerbSynset"

        # Check for adjective-specific relations
        elif "similar" in clean_relation or "satellite" in clean_relation:
            return "AdjectiveSynset", relation, "AdjectiveSynset"

    # Default fallback for WordNet
    return "Synset", relation, "Synset"


def extract_freebase_domain_range(relation):
    """
    Extract domain and range types directly from Freebase relation path structure.
    For relations with _inv suffix, domain and range are swapped.
    """
    # Check if this is an inverse relation
    is_inverse = relation.endswith("_inv")

    # Remove _inv suffix for processing if present
    base_relation = relation[:-4] if is_inverse else relation

    # Initialize default types
    domain_type = "Entity"
    range_type = "Entity"

    # Handle complex relations with dotted notation
    if "./" in base_relation:
        # Split into first and second hop
        first_hop, second_hop = base_relation.split("./", 1)

        # Extract domain from first hop - second component
        if first_hop.startswith('/') and first_hop.count('/') >= 2:
            components = first_hop.strip('/').split('/')
            if len(components) >= 2:
                domain_type = components[1]

        # Extract range from second hop - last component
        if second_hop.count('/') >= 1:
            components = second_hop.strip('/').split('/')
            if len(components) >= 1:
                range_type = components[-1]
    # Handle simple relations without dots
    else:
        if base_relation.startswith('/') and base_relation.count('/') >= 2:
            components = base_relation.strip('/').split('/')
            if len(components) >= 2:
                domain_type = components[1]
                if len(components) >= 3:
                    range_type = components[2]

    # Format types (remove trailing _s from plurals)
    if domain_type != "Entity" and not domain_type.isspace() and domain_type.endswith('_s'):
        domain_type = domain_type[:-2]
    if range_type != "Entity" and not range_type.isspace() and range_type.endswith('_s'):
        range_type = range_type[:-2]

    # For inverse relations, swap domain and range
    if is_inverse:
        domain_type, range_type = range_type, domain_type

    return domain_type, relation, range_type


def extract_relations_from_triples(triples_file):
    """Extract unique relations from triples file."""
    relations = set()
    with open(triples_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                relations.add(parts[1])
    return sorted(relations)


def calculate_pattern_importance(pattern, domain_entities, range_entities, dataset_type="generic"):
    """Calculate a meaningful importance value for a schema pattern."""
    domain_type, relation_type, range_type = pattern

    # Factor 1: Specific vs. General Types
    specificity = 1.0
    if domain_type != "Entity" and range_type != "Entity":
        specificity = 1.8
    elif domain_type != "Entity" or range_type != "Entity":
        specificity = 1.4

    # Factor 2: Entity Count
    domain_count = len(domain_entities) if domain_entities else 0
    range_count = len(range_entities) if range_entities else 0

    entity_factor = 1.0
    if domain_count > 0 and range_count > 0:
        avg_count = (domain_count + range_count) / 2
        if avg_count < 10:
            entity_factor = 0.8
        elif avg_count < 100:
            entity_factor = 1.2
        elif avg_count < 1000:
            entity_factor = 1.5
        else:
            entity_factor = 1.8

    # Factor 3: Relation Semantics
    semantic_factor = 1.0

    # Dataset-specific semantic importance
    if dataset_type == "wordnet":
        # WordNet-specific important relations
        if any(rel in relation_type for rel in ["hypernym", "hyponym"]):
            semantic_factor = 1.8  # Very important hierarchical relations
        elif any(rel in relation_type for rel in ["meronym", "holonym"]):
            semantic_factor = 1.6  # Part-whole relations
        elif "domain" in relation_type:
            semantic_factor = 1.4  # Domain relations
        elif any(rel in relation_type for rel in ["similar", "antonym"]):
            semantic_factor = 1.3  # Similarity/opposition relations
    else:
        # Freebase/generic semantic importance
        hierarchical_keywords = ['type', 'subclass', 'instance', 'is_a', 'kind_of', 'category']
        if any(kw in relation_type.lower() for kw in hierarchical_keywords):
            semantic_factor = 1.6

        core_types = ['person', 'organization', 'event', 'location', 'concept']
        if any(t in domain_type.lower() for t in core_types) or any(t in range_type.lower() for t in core_types):
            semantic_factor = max(semantic_factor, 1.4)

    # Factor 4: Relation complexity
    complexity_factor = 1.0
    if "./" in relation_type:
        complexity_factor = 1.3  # Complex join relation (Freebase)
    elif "_" in relation_type and relation_type.count("_") > 2:
        complexity_factor = 1.2  # Complex WordNet relation

    # Combine all factors
    importance = (
            0.30 * specificity +
            0.25 * entity_factor +
            0.25 * semantic_factor +
            0.20 * complexity_factor
    )

    # Ensure importance is in range [0.5, 2.5]
    importance = max(0.5, min(2.5, importance))
    return round(importance, 2)


def build_schema(dataset: str = DEFAULT_DATASET):
    """Build schema for the specified dataset folder."""

    # Detect dataset type from the dataset argument
    dataset_type = "wordnet" if "wn" in dataset.lower() else "freebase"

    # Construct paths
    base_dir = os.path.join(BASE_DIR, dataset)
    triples_file = os.path.join(base_dir, "triples.txt")
    relation_names_file = os.path.join(base_dir, "relation_names.txt")
    schema_file = os.path.join(base_dir, "schema.txt")
    schema_store_file = os.path.join(base_dir, "schema_store.json")
    entity_types_file = os.path.join(base_dir, "entity_types.json")
    entity_mappings_file = os.path.join(base_dir, "entity_mappings.json")
    relation_mappings_file = os.path.join(base_dir, "relation_mappings.json")

    print(f"Building schema for {dataset_type} dataset: {dataset}")
    print(f"Reading from: {triples_file}")

    # Check if triples file exists
    if not os.path.exists(triples_file):
        print(f"Error: Triples file not found at {triples_file}")
        print(f"Current working directory: {os.getcwd()}")
        if os.path.isdir(base_dir):
            print(f"Contents of {base_dir}:")
            for item in os.listdir(base_dir):
                print(f"  - {item}")
        else:
            print(f"Dataset directory {base_dir} does not exist.")
        return

    # Extract relations from triples if relation_names.txt doesn't exist
    if not os.path.exists(relation_names_file):
        print(f"Extracting relations from {triples_file}...")
        relations = extract_relations_from_triples(triples_file)

        # Write relations to file
        with open(relation_names_file, 'w', encoding='utf-8') as f:
            for relation in relations:
                f.write(relation + "\n")
        print(f"Saved {len(relations)} relations to {relation_names_file}")
    else:
        # Load existing relations
        with open(relation_names_file, 'r', encoding='utf-8') as f:
            relations = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(relations)} relations from {relation_names_file}")

    # Build relation mappings
    relation2id = {rel: i for i, rel in enumerate(sorted(relations))}
    id2relation = {i: rel for rel, i in relation2id.items()}

    # Extract schema patterns
    schema_patterns = []
    relation_to_types = {}

    print(f"\nExtracting schema patterns for {dataset_type} relations...")
    for i, relation in enumerate(relations):
        domain_type, relation_type, range_type = extract_domain_range_from_path(relation, dataset_type)
        relation_to_types[relation] = (domain_type, range_type)
        schema_triple = f"{domain_type} {relation} {range_type}"
        schema_patterns.append(schema_triple)

        # Show first few patterns as examples
        if i < 5:
            print(f"  {relation} -> {domain_type} -- {range_type}")

    # Write schema to file
    with open(schema_file, 'w', encoding='utf-8') as f:
        for pattern in schema_patterns:
            f.write(pattern + "\n")
    print(f"\nSchema saved to {schema_file} with {len(schema_patterns)} patterns")

    # Save relation mappings
    with open(relation_mappings_file, 'w', encoding='utf-8') as f:
        json.dump({
            'dataset_type': dataset_type,
            'relation_to_types': {k: list(v) for k, v in relation_to_types.items()},
            'relation_count': len(relation_to_types),
            'relation_to_id': relation2id,
            'id_to_relation': {str(k): v for k, v in id2relation.items()}
        }, f, indent=2, ensure_ascii=False)
    print(f"Relation mappings saved to {relation_mappings_file}")

    # Build entity information from triples
    print(f"\nProcessing entities from {triples_file}...")

    entity_to_id = {}
    id_to_entity = {}
    entity_types = defaultdict(set)
    entity_counter = 0

    # Process triples to extract entities and their types
    triple_count = 0
    with open(triples_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                head, relation, tail = parts
                triple_count += 1

                # Add entities to mapping
                if head not in entity_to_id:
                    entity_to_id[head] = entity_counter
                    id_to_entity[entity_counter] = head
                    entity_counter += 1

                if tail not in entity_to_id:
                    entity_to_id[tail] = entity_counter
                    id_to_entity[entity_counter] = tail
                    entity_counter += 1

                # Assign types based on relation
                if relation in relation_to_types:
                    domain_type, range_type = relation_to_types[relation]
                    entity_types[head].add(domain_type)
                    entity_types[tail].add(range_type)

    print(f"Processed {triple_count} triples")
    print(f"Found {len(entity_to_id)} unique entities")

    # Save entity types
    with open(entity_types_file, 'w', encoding='utf-8') as f:
        json.dump({k: list(v) for k, v in entity_types.items()}, f, indent=2)
    print(f"Entity types saved to {entity_types_file}")

    # Save entity mappings
    with open(entity_mappings_file, 'w', encoding='utf-8') as f:
        json.dump({
            'entity_to_id': entity_to_id,
            'id_to_entity': {str(k): v for k, v in id_to_entity.items()},
            'entity_count': len(entity_to_id)
        }, f, indent=2)
    print(f"Entity mappings saved to {entity_mappings_file}")

    # Build schema store
    print("\nBuilding schema store...")

    # Map entities to their types
    type_to_entities = defaultdict(set)
    for entity, types in entity_types.items():
        entity_id = entity_to_id[entity]
        for type_name in types:
            type_to_entities[type_name].add(entity_id)

    # Show type statistics
    print(f"\nEntity type statistics:")
    sorted_types = sorted([(t, len(e)) for t, e in type_to_entities.items()],
                          key=lambda x: x[1], reverse=True)
    for type_name, count in sorted_types[:10]:
        print(f"  {type_name}: {count} entities")

    # Build schema store with importance values
    schema_store = {}
    relation_to_pattern = {}

    print("\nCalculating pattern importance values...")
    for pattern in schema_patterns:
        parts = pattern.split(' ', 2)
        if len(parts) >= 3:
            domain, relation, range_ = parts
            pattern_key = (domain, relation, range_)

            # Get entities of domain and range types
            domain_entities = type_to_entities.get(domain, set())
            range_entities = type_to_entities.get(range_, set())

            # Calculate importance
            importance = calculate_pattern_importance(pattern_key, domain_entities, range_entities, dataset_type)

            # Add to schema store
            schema_store[pattern_key] = {
                'importance': importance,
                'entities': {
                    'head': list(domain_entities),
                    'tail': list(range_entities)
                },
                'relations': [relation]
            }

            relation_to_pattern[relation] = pattern_key

    # Save schema store
    serializable_store = {}
    for pattern, data in schema_store.items():
        pattern_str = str(pattern)
        serializable_store[pattern_str] = data

    with open(schema_store_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_store, f, indent=2)
    print(f"Schema store saved to {schema_store_file} with {len(serializable_store)} patterns")

    # Create detailed schema file
    detailed_schema_file = os.path.join(base_dir, "detailed_schema.txt")
    with open(detailed_schema_file, 'w', encoding='utf-8') as f:
        f.write(f"# Detailed Schema for {dataset} ({dataset_type})\n")
        f.write("# Format: Domain Relation Range [Importance]\n")
        f.write("#" + "=" * 60 + "\n\n")

        # Sort patterns by importance for better readability
        sorted_patterns = sorted(schema_patterns,
                                 key=lambda p: schema_store.get(tuple(p.split(' ', 2)), {}).get('importance', 0),
                                 reverse=True)

        for pattern in sorted_patterns:
            parts = pattern.split(' ', 2)
            if len(parts) >= 3:
                domain, relation, range_ = parts
                pattern_key = (domain, relation, range_)

                if pattern_key in schema_store:
                    importance = schema_store[pattern_key]['importance']
                    head_count = len(schema_store[pattern_key]['entities']['head'])
                    tail_count = len(schema_store[pattern_key]['entities']['tail'])
                    f.write(f"{domain} {relation} {range_} [{importance:.2f}] (head:{head_count}, tail:{tail_count})\n")

    print(f"Detailed schema saved to {detailed_schema_file}")

    # Save relation-to-pattern mapping
    relation_pattern_file = os.path.join(base_dir, "relation_to_pattern.json")
    with open(relation_pattern_file, 'w', encoding='utf-8') as f:
        serializable_mapping = {
            rel: str(pattern) for rel, pattern in relation_to_pattern.items()
        }
        json.dump(serializable_mapping, f, indent=2)
    print(f"Relation-to-pattern mapping saved to {relation_pattern_file}")

    print("\n" + "=" * 60)
    print("Schema generation complete!")
    print(f"Dataset type: {dataset_type}")
    print(f"Total relations: {len(relations)}")
    print(f"Total entities: {len(entity_to_id)}")
    print(f"Total entity types: {len(type_to_entities)}")
    print("=" * 60)


def _build_cli_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate schema metadata files for a knowledge graph dataset."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Dataset folder inside data/ (e.g. fb15k-237-10, fb15k-237-20, wn18rr-10, wn18rr-20)",
    )
    return parser


if __name__ == "__main__":
    cli_parser = _build_cli_parser()
    cli_args = cli_parser.parse_args()
    build_schema(dataset=cli_args.dataset)
