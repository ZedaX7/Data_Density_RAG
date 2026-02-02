import faiss
import json
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple

# ==================== CONFIGURATION ====================
BASE_PATH = "./data/faiss_encoded/"
DEVICE = "cuda:0"
CACHE_FOLDER = "./pre_trained/hf_cache"

# Hybrid scoring weights (should sum to 1.0)
WEIGHT_EMBEDDING = 0.3    # Semantic similarity weight
WEIGHT_FEATURE = 0.7      # Feature distance weight

# Constraint tolerance for pre-filtering (as fraction of target value)
CONSTRAINT_TOLERANCE = 0.5  # Allow 50% deviation for pre-filtering
# =======================================================

# Global cache for loaded models and indices
_cache = {}


def parse_query_constraints(query: str) -> Dict[str, Optional[float]]:
    """
    Parse numerical constraints from a natural language query.

    Extracts voltage, capacity, and dimension constraints from queries like:
    "I need a 14.8V and 10Ah battery pack under 120mm × 60mm × 40mm"

    Returns:
        Dict with keys: voltage, capacity, width_mm, depth_mm, height_mm
        Values are None if not found in query.
    """
    constraints = {
        "voltage": None,
        "capacity": None,
        "width_mm": None,
        "depth_mm": None,
        "height_mm": None,
    }

    # Voltage patterns: "14.8V", "14.8 V", "14.8 volts"
    voltage_match = re.search(r'(\d+\.?\d*)\s*[Vv](?:olts?)?(?!\w)', query)
    if voltage_match:
        constraints["voltage"] = float(voltage_match.group(1))

    # Capacity patterns: "10Ah", "10 Ah", "10 amp-hours"
    capacity_match = re.search(r'(\d+\.?\d*)\s*[Aa]h', query)
    if capacity_match:
        constraints["capacity"] = float(capacity_match.group(1))

    # Dimension patterns: "120mm × 60mm × 40mm" or "120x60x40mm" or "120 x 60 x 40"
    # Also handles "under 120mm × 60mm × 40mm"
    dim_pattern = r'(\d+\.?\d*)\s*(?:mm)?\s*[×xX]\s*(\d+\.?\d*)\s*(?:mm)?\s*[×xX]\s*(\d+\.?\d*)\s*(?:mm)?'
    dim_match = re.search(dim_pattern, query)
    if dim_match:
        constraints["width_mm"] = float(dim_match.group(1))
        constraints["depth_mm"] = float(dim_match.group(2))
        constraints["height_mm"] = float(dim_match.group(3))

    return constraints


def load_retrieval_data(data_name: str) -> Tuple:
    """
    Load and cache all retrieval data (index, metadata, features, config).
    """
    cache_key = f"data_{data_name}"

    if cache_key not in _cache:
        path = f"{BASE_PATH}{data_name}/"

        # Load FAISS index
        index = faiss.read_index(f"{path}rag_index_{data_name}.faiss")

        # Load feature matrix
        features = np.load(f"{path}rag_features_{data_name}.npy")

        # Load metadata
        with open(f"{path}rag_metadata_{data_name}.json") as f:
            metadata = json.load(f)

        # Load config (includes feature stats)
        with open(f"{path}rag_config_{data_name}.json") as f:
            config = json.load(f)

        _cache[cache_key] = (index, features, metadata, config)

    return _cache[cache_key]


def load_embedding_model(model_name: str) -> SentenceTransformer:
    """Load and cache the embedding model."""
    cache_key = f"model_{model_name}"

    if cache_key not in _cache:
        _cache[cache_key] = SentenceTransformer(
            model_name,
            device=DEVICE,
            cache_folder=CACHE_FOLDER
        )

    return _cache[cache_key]


def filter_by_constraints(
    features: np.ndarray,
    constraints: Dict[str, Optional[float]],
    metadata: List[Dict],
    config: Dict,
    tolerance: float = CONSTRAINT_TOLERANCE
) -> Tuple[List[int], np.ndarray]:
    """
    Pre-filter designs by numerical constraints.

    Returns indices of designs that satisfy constraints within tolerance.
    """
    feature_keys = config["feature_keys"]
    n_designs = len(metadata)

    # Map constraint keys to feature keys
    constraint_to_feature = {
        "voltage": "nominal_voltage",
        "capacity": "capacity_ah",
        "width_mm": "physical_width_mm",
        "depth_mm": "physical_depth_mm",
        "height_mm": "physical_height_mm",
    }

    valid_mask = np.ones(n_designs, dtype=bool)

    for constraint_key, target_value in constraints.items():
        if target_value is None:
            continue

        feature_key = constraint_to_feature.get(constraint_key)
        if feature_key is None or feature_key not in feature_keys:
            continue

        feature_idx = feature_keys.index(feature_key)
        feature_values = features[:, feature_idx]

        # For dimensions, filter designs that fit WITHIN the constraint
        # (design should be smaller than or equal to target)
        if constraint_key in ["width_mm", "depth_mm", "height_mm"]:
            # Allow some tolerance above target for dimensions
            max_allowed = target_value * (1 + tolerance)
            valid_mask &= (feature_values <= max_allowed)
        else:
            # For voltage/capacity, look for designs within tolerance range
            min_allowed = target_value * (1 - tolerance)
            max_allowed = target_value * (1 + tolerance)
            valid_mask &= (feature_values >= min_allowed) & (feature_values <= max_allowed)

    valid_indices = np.where(valid_mask)[0].tolist()

    return valid_indices, features[valid_mask] if valid_indices else features


def compute_feature_distance(
    target_constraints: Dict[str, Optional[float]],
    features: np.ndarray,
    config: Dict,
    indices: Optional[List[int]] = None
) -> np.ndarray:
    """
    Compute weighted normalized feature distance for each design.

    Lower distance = better match to target constraints.
    Returns similarity scores (1 / (1 + distance)) so higher = better.
    """
    feature_keys = config["feature_keys"]
    feature_stats = config["feature_stats"]

    # Map constraint keys to feature keys
    constraint_to_feature = {
        "voltage": "nominal_voltage",
        "capacity": "capacity_ah",
        "width_mm": "physical_width_mm",
        "depth_mm": "physical_depth_mm",
        "height_mm": "physical_height_mm",
    }

    if indices is not None:
        features_subset = features[indices]
    else:
        features_subset = features

    n_designs = features_subset.shape[0]
    distances = np.zeros(n_designs, dtype=np.float32)
    total_weight = 0.0

    for constraint_key, target_value in target_constraints.items():
        if target_value is None:
            continue

        feature_key = constraint_to_feature.get(constraint_key)
        if feature_key is None or feature_key not in feature_keys:
            continue

        feature_idx = feature_keys.index(feature_key)
        stats = feature_stats[feature_key]
        weight = stats["weight"]

        # Normalize by range
        range_val = stats["max"] - stats["min"]
        if range_val == 0:
            range_val = 1.0

        # Compute normalized absolute difference
        feature_values = features_subset[:, feature_idx]
        normalized_diff = np.abs(feature_values - target_value) / range_val

        distances += weight * normalized_diff
        total_weight += weight

    # Normalize by total weight
    if total_weight > 0:
        distances /= total_weight

    # Convert distance to similarity (higher = better)
    similarity = 1.0 / (1.0 + distances)

    return similarity


def compute_embedding_similarity(
    query: str,
    index: faiss.Index,
    model: SentenceTransformer,
    indices: Optional[List[int]] = None,
    top_k: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute embedding similarity scores using FAISS index.

    Returns (scores, indices) for top_k matches.
    """
    # Encode query
    query_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    if indices is not None and len(indices) > 0:
        # Search only within filtered indices
        # For small filtered sets, just search all and filter
        search_k = min(top_k * 10, index.ntotal)
        scores, result_indices = index.search(query_vec, search_k)
        scores = scores[0]
        result_indices = result_indices[0]

        # Filter to only include valid indices
        indices_set = set(indices)
        mask = [i in indices_set for i in result_indices]
        scores = scores[mask][:top_k]
        result_indices = result_indices[mask][:top_k]
    else:
        scores, result_indices = index.search(query_vec, top_k)
        scores = scores[0]
        result_indices = result_indices[0]

    return scores, result_indices


def retrieve_relevant_docs(
    query: str,
    data_name: str = "full_[64-64-4-64-64]",
    top_k: int = 5,
    use_hybrid: bool = True,
    embedding_weight: float = WEIGHT_EMBEDDING,
    feature_weight: float = WEIGHT_FEATURE,
    constraint_tolerance: float = CONSTRAINT_TOLERANCE,
    verbose: bool = False
) -> List[Dict]:

    # Load all necessary data
    index, features, metadata, config = load_retrieval_data(data_name)
    model = load_embedding_model(config["embedding_model"])

    if verbose:
        print(f"[Retrieval] Loaded {len(metadata)} designs from {data_name}")

    # Parse constraints from query
    constraints = parse_query_constraints(query)

    if verbose:
        print(f"[Retrieval] Parsed constraints: {constraints}")

    if not use_hybrid:
        # Pure embedding-based retrieval (legacy behavior)
        scores, result_indices = compute_embedding_similarity(
            query, index, model, top_k=top_k
        )
        return [metadata[i] for i in result_indices]

    # Hybrid retrieval
    # Step 1: Pre-filter by constraints
    valid_indices, filtered_features = filter_by_constraints(
        features, constraints, metadata, config, constraint_tolerance
    )

    if verbose:
        print(f"[Retrieval] Pre-filtered to {len(valid_indices)} candidates")

    # If no designs pass filter, relax and use all
    if len(valid_indices) == 0:
        if verbose:
            print("[Retrieval] No designs match constraints, using all designs")
        valid_indices = list(range(len(metadata)))

    # Step 2: Compute feature-based similarity for filtered designs
    feature_scores = compute_feature_distance(
        constraints, features, config, valid_indices
    )

    # Step 3: Compute embedding similarity for filtered designs
    embedding_scores, emb_indices = compute_embedding_similarity(
        query, index, model, valid_indices, top_k=len(valid_indices)
    )

    # Map embedding scores back to valid_indices order
    emb_score_map = {idx: score for idx, score in zip(emb_indices, embedding_scores)}
    embedding_scores_aligned = np.array([
        emb_score_map.get(idx, 0.0) for idx in valid_indices
    ], dtype=np.float32)

    # Normalize scores to [0, 1]
    if embedding_scores_aligned.max() > embedding_scores_aligned.min():
        embedding_scores_aligned = (embedding_scores_aligned - embedding_scores_aligned.min()) / \
                                    (embedding_scores_aligned.max() - embedding_scores_aligned.min())

    # Step 4: Combine scores
    hybrid_scores = (embedding_weight * embedding_scores_aligned + feature_weight * feature_scores)

    # Step 5: Get top_k by hybrid score
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
    result_indices = [valid_indices[i] for i in top_indices]

    if verbose:
        print(f"[Retrieval] Top {top_k} results:")
        for rank, idx in enumerate(result_indices):
            feat = metadata[idx]["features"]
            print(f"  {rank+1}. {feat['nominal_voltage']}V, {feat['capacity_ah']}Ah, "
                  f"{feat['physical_width_mm']}×{feat['physical_depth_mm']}×{feat['physical_height_mm']}mm "
                  f"(hybrid_score={hybrid_scores[top_indices[rank]]:.3f})")

    return [metadata[i] for i in result_indices]


# Convenience function for testing
def test_retrieval():
    """Test the retrieval system with a sample query."""
    query = "I need a 14.8V and 10Ah battery pack under 120mm × 60mm × 80mm, using 18650 cells."

    print("=" * 60)
    print("Testing Hybrid Retrieval")
    print("=" * 60)
    print(f"Query: {query}\n")

    results = retrieve_relevant_docs(
        query,
        data_name="full_[16-16-4-16-16]",
        top_k=5,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("Retrieved Designs:")
    print("=" * 60)
    for i, doc in enumerate(results):
        f = doc["features"]
        print(f"\n{i+1}. Configuration: {f['series_count']}S{f['parallel_count']}P")
        print(f"   Voltage: {f['nominal_voltage']}V, Capacity: {f['capacity_ah']}Ah")
        print(f"   Dimensions: {f['physical_width_mm']}×{f['physical_depth_mm']}×{f['physical_height_mm']}mm")
        print(f"   Total cells: {f['total_cells']}")


if __name__ == "__main__":
    test_retrieval()
