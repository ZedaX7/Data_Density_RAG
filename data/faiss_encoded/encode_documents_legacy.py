import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

sub_set = "full"
raw_set = "[16-16-4-16-16]"

# 1. Load the new battery pack dataset (.jsonl format)
designs = []
with open("./data/raw/" + sub_set + "_" + raw_set + "/enumerated_battery_pack_dataset_" + raw_set + ".json", "r") as f:
    designs = json.load(f)

# 2. Format each design entry into a retrievable string
def format_design_entry(entry):
    f = entry.get("features", {})  # Extract 'features' dictionary

    return (
        f"Configuration: {f.get('configuration_type', 'N/A')} | "
        f"Series: {f.get('series_count', 'N/A')}, "
        f"Parallel: {f.get('parallel_count', 'N/A')} | "
        f"Total Cells: {f.get('total_cells', 'N/A')} | "
        f"Grid Dimensions: {f.get('num_cells_width', 'N/A')}×"
        f"{f.get('num_cells_depth', 'N/A')}×"
        f"{f.get('num_cells_height', 'N/A')} | "
        f"Voltage: {f.get('nominal_voltage', 'N/A')}V, "
        f"Capacity: {f.get('capacity_ah', 'N/A')}Ah, "
        f"Energy: {f.get('energy_wh', 'N/A')}Wh | "
        f"Physical Dimensions: {f.get('physical_width_mm', 'N/A')}×"
        f"{f.get('physical_depth_mm', 'N/A')}×"
        f"{f.get('physical_height_mm', 'N/A')}mm | "
        f"Weight: {f.get('weight_kg', 'N/A')}kg | "
        f"Energy Density: {f.get('energy_density_wh_kg', 'N/A')} Wh/kg | "
        f"Internal Resistance: {f.get('internal_resistance_ohm', 'N/A')} Ω | "
        f"Voltage Range: {f.get('min_voltage', 'N/A')}V–{f.get('max_voltage', 'N/A')}V | "
        f"Max Discharge: {f.get('max_discharge_current', 'N/A')}A | "
        f"Max Charge: {f.get('max_charge_current', 'N/A')}A"
    )

# Build corpus for embedding and full metadata ===
descriptions = []
metadata_list = []

for d in designs:
    text = format_design_entry(d)
    descriptions.append(text)
    metadata_list.append({
        "features": d.get("features", {}),
        "cell_locations": d.get("cell_locations", []),
        # "pybamm_validated": d.get("pybamm_validated", False),
        # "generation_method": d.get("generation_method", ""),
        # "description": text
    })


# 3. Embed with Sentence-BERT
# "all-mpnet-base-v2"，"multi-qa-mpnet-base-dot-v1", "all-MiniLM-L12-v2", "all-MiniLM-L6-v2"
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer("sentence-transformers/" + model_name)  # or domain-tuned version
embeddings = model.encode(descriptions, convert_to_numpy=True)

# 4. Build & save FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

output_dir = "./data/faiss_encoded/" + sub_set + "_" + raw_set
os.makedirs(output_dir, exist_ok=True)

faiss.write_index(index, output_dir + "/rag_index_" + sub_set + "_" + raw_set + ".faiss")

# Save metadata for lookup
with open(output_dir + "/rag_metadata_" + sub_set + "_" + raw_set + ".json", "w") as f:
    json.dump(metadata_list, f)
