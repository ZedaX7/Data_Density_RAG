"""
Hybrid Encoding for Battery Pack RAG System

This module creates embeddings and feature indices for hybrid retrieval:
1. Dense embeddings using BGE-large for semantic similarity
2. Raw numerical features for constraint-based filtering and feature distance
"""

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

sub_set = "full"
raw_set = "[16-16-4-16-16]"

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
# Alternatives: "intfloat/e5-large-v2", "BAAI/bge-base-en-v1.5" (smaller)

# Features to extract for hybrid retrieval (with importance weights)
FEATURE_CONFIG = {
    "nominal_voltage": {"weight": 2.0, "type": "numeric"},
    "capacity_ah": {"weight": 2.0, "type": "numeric"},
    "energy_wh": {"weight": 1.5, "type": "numeric"},
    "total_cells": {"weight": 1.0, "type": "numeric"},
    "series_count": {"weight": 1.5, "type": "numeric"},
    "parallel_count": {"weight": 1.5, "type": "numeric"},
    "physical_width_mm": {"weight": 1.0, "type": "numeric"},
    "physical_depth_mm": {"weight": 1.0, "type": "numeric"},
    "physical_height_mm": {"weight": 1.0, "type": "numeric"},
    "weight_kg": {"weight": 0.5, "type": "numeric"},
    # "energy_density_wh_kg": {"weight": 0.5, "type": "numeric"},
    "num_cells_width": {"weight": 0.5, "type": "numeric"},
    "num_cells_depth": {"weight": 0.5, "type": "numeric"},
    "num_cells_height": {"weight": 0.5, "type": "numeric"},
}


def format_design_for_embedding(entry):
    """
    Format design entry into natural language for embedding.
    Uses query-aligned phrasing for better retrieval.
    """
    f = entry.get("features", {})

    return (
        f"Battery pack with {f.get('nominal_voltage', 0):.1f}V nominal voltage "
        f"and {f.get('capacity_ah', 0):.1f}Ah capacity providing {f.get('energy_wh', 0):.1f}Wh energy. "
        f"Uses {f.get('total_cells', 0)} cells in {f.get('series_count', 0)}S{f.get('parallel_count', 0)}P configuration. "
        f"Grid layout: {f.get('num_cells_width', 0)}×{f.get('num_cells_depth', 0)}×{f.get('num_cells_height', 0)} cells. "
        f"Physical dimensions: {f.get('physical_width_mm', 0):.0f}mm width × "
        f"{f.get('physical_depth_mm', 0):.0f}mm depth × {f.get('physical_height_mm', 0):.0f}mm height. "
        f"Weight: {f.get('weight_kg', 0):.3f}kg. " # with {f.get('energy_density_wh_kg', 0):.1f} Wh/kg energy density. "
        # f"18650 lithium-ion cells with {f.get('internal_resistance_ohm', 0):.3f} ohm internal resistance."
    )


def extract_features(entry):
    """Extract numerical features from a design entry."""
    f = entry.get("features", {})
    return {key: f.get(key, 0.0) for key in FEATURE_CONFIG.keys()}


def compute_feature_statistics(all_features):
    """Compute min, max, mean, std for each feature for normalization."""
    stats = {}
    for key in FEATURE_CONFIG.keys():
        values = [f[key] for f in all_features]
        stats[key] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)) if np.std(values) > 0 else 1.0,
            "weight": FEATURE_CONFIG[key]["weight"]
        }
    return stats


def main():
    print(f"[INFO] Loading dataset: {sub_set}_{raw_set}")

    # 1. Load the battery pack dataset
    input_path = f"./data/raw/{sub_set}_{raw_set}/enumerated_battery_pack_dataset_{raw_set}.json"
    with open(input_path, "r") as f:
        designs = json.load(f)

    print(f"[INFO] Loaded {len(designs)} designs")

    # 2. Prepare data for encoding
    descriptions = [] # Natural language text for embedding
    feature_list = [] # Just the numerical features for hybrid filtering
    metadata_list = [] # Full design info (features + cell locations) for returning to users

    for d in designs:
        # Text description for embedding
        text_description = format_design_for_embedding(d)
        descriptions.append(text_description)

        # Numerical features for hybrid retrieval
        feature = extract_features(d)
        feature_list.append(feature)

        # Full metadata for retrieval results
        metadata_list.append({
            "features": d.get("features", {}),
            "cell_locations": d.get("cell_locations", []),
        })

    # 3. Compute feature statistics for normalization
    feature_stats = compute_feature_statistics(feature_list)
    print(f"[INFO] Computed feature statistics for {len(feature_stats)} features")

    # 4. Create feature matrix (for fast vectorized distance calculations/numpy operations during retrieval)
    feature_keys = list(FEATURE_CONFIG.keys()) # keys
    feature_matrix = np.array([
        [f[key] for key in feature_keys] for f in feature_list
    ], dtype=np.float32)

    # 5. Load embedding model and encode all text descriptions into dense vectors
    print(f"[INFO] Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"[INFO] Encoding {len(descriptions)} descriptions...")
    embeddings = model.encode(
        descriptions,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True  # Normalize for cosine similarity
    )
    print(f"[INFO] Embedding dimension: {embeddings.shape[1]}")

    # 6. Build FAISS index with Inner Product (cosine similarity for normalized vectors)
    index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner Product = cosine for normalized
    index.add(embeddings)
    print(f"[INFO] Built FAISS index with {index.ntotal} vectors")

    # 7. Save everything
    output_dir = f"./data/faiss_encoded/{sub_set}_{raw_set}"
    os.makedirs(output_dir, exist_ok=True)

    # Save FAISS index
    index_path = f"{output_dir}/rag_index_{sub_set}_{raw_set}.faiss"
    faiss.write_index(index, index_path)
    print(f"[INFO] Saved FAISS index to: {index_path}")

    # Save feature matrix
    features_path = f"{output_dir}/rag_features_{sub_set}_{raw_set}.npy"
    np.save(features_path, feature_matrix)
    print(f"[INFO] Saved feature matrix to: {features_path}")

    # Save metadata
    metadata_path = f"{output_dir}/rag_metadata_{sub_set}_{raw_set}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata_list, f)
    print(f"[INFO] Saved metadata to: {metadata_path}")

    # Save feature statistics and config
    config_path = f"{output_dir}/rag_config_{sub_set}_{raw_set}.json"
    with open(config_path, "w") as f:
        json.dump({
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": int(embeddings.shape[1]),
            "num_designs": len(designs),
            "feature_keys": feature_keys,
            "feature_stats": feature_stats,
            "feature_config": FEATURE_CONFIG,
        }, f, indent=2)
    print(f"[INFO] Saved config to: {config_path}")

    print(f"\n[SUCCESS] Encoding complete!")
    print(f"  - Designs indexed: {len(designs)}")
    print(f"  - Embedding model: {EMBEDDING_MODEL}")
    print(f"  - Embedding dimension: {embeddings.shape[1]}")
    print(f"  - Features tracked: {len(feature_keys)}")


if __name__ == "__main__":
    main()
