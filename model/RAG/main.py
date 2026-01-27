import os
import sys
import json
from datetime import datetime
import random
import pandas as pd
import numpy as np
import torch

from retrieve import retrieve_relevant_docs
from generate import generate_answer, initialize_model
from visual_rendering import render_battery_pack
from validation import (extract_cell_locations, extract_cell_connections, extract_design_features, 
                        validate_with_pybamm, validate_design, print_validation_summary, ValidationResult)

# ==================== USER CONFIGURATION ====================
# Data configuration
sub_set = "full"
raw_set = "[32-32-4-32-32]"
DATA_NAME = sub_set + "_" + raw_set

# Model configuration
# Available models: "llama3-8b", "llama31-8b", "llama32-3b", "llama33-70b", "llama4-17b"
MODEL_KEY = "llama32-3b"
# ============================================================

def generate_timestamped_basename(prefix="battery_pack_rendering"):
    """Generate a shared basename using timestamp for HTML and JSON files."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    return f"{prefix}_{timestamp}"


def generate_random_prompt(seed=None, cell_limit=16):
    """Generate a random battery pack design prompt with specified seed."""
    if seed is not None:
        random.seed(seed)

    # Random features
    voltages_options = [round(3.7 * i, 1) for i in range(1, cell_limit + 1)] # 3.7V * # of cells in series
    capacity_options = [round(2.5 * i, 1) for i in range(1, cell_limit + 1)] # 2.5Ah * # of cells in parallel

    width_options = [round(10 + 20 * i) for i in range(1, cell_limit + 1)] # 20mm * # of cells in width, 10 mm extra for spacing
    depth_options = [round(10 + 20 * i) for i in range(1, cell_limit + 1)] # 20mm * # of cells in depth, 10 mm extra for spacing
    height_options = [round(10 + 65 * i) for i in range(1, cell_limit + 1)] # 65mm * # of cells in height, 10 mm extra for spacing

    # Random applications
    applications = [
        "high current drones",
        "electric vehicles",
        "power tools",
        "energy storage systems",
        "robotics",
        "portable electronics",
        "electric bikes",
        "marine applications"
    ]

    voltage = random.choice(voltages_options)
    capacity = random.choice(capacity_options)
    width = random.choice(width_options)
    depth = random.choice(depth_options)
    height = random.choice(height_options)
    # application = random.choice(applications)

    prompt = f"I need a {voltage}V, {capacity}Ah battery pack under {width}mm × {depth}mm × {height}mm, using 18650 cells."
    # prompt = f"I need a {voltage}V battery pack under {width}mm × {depth}mm × {height}mm, optimized for {application}, using 18650 cells."

    return prompt, voltage, capacity, width, depth, height #, application


def main(query, required_specs=None, render=False):
    """
    Generate a single battery pack design for the given query.

    Args:
        query: The design prompt
        required_specs: Optional dict with keys: voltage, capacity, width_mm, depth_mm, height_mm
        render: Whether to render the battery pack visualization
    """
    docs = retrieve_relevant_docs(query, data_name=DATA_NAME, top_k=3)
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        output = generate_answer(query, docs, model_key=MODEL_KEY)
        print(output)

        # Perform comprehensive validation
        print("\n" + "="*50)
        print("PERFORMING COMPREHENSIVE VALIDATION")
        print("="*50)
        validation_results = validate_design(
            output=output,
            required_specs=required_specs
        )

        # pybamm_result = validate_with_pybamm(features)
        all_valid = print_validation_summary(validation_results)
        if all_valid:
            cell_locations = extract_cell_locations(output)
            features = extract_design_features(output)
            print("✅ Design validated successfully.")
            break
        else:
            print("[WARNING] Design invalid; retrying generation...")
            retry_count += 1
            continue

    # If max retries reached, return None
    if retry_count >= max_retries:
        print(f"[ERROR] Failed to generate valid design after {max_retries} retries.")
        return None

    # Save outputs
    os.makedirs("output", exist_ok=True)

    basename = generate_timestamped_basename()
    html_path = os.path.join("output", f"{basename}.html")
    json_path = os.path.join("output", f"{basename}.json")

    if render:
        print(f"[INFO] Rendering battery pack to: {html_path}")
        render_battery_pack(cell_locations, output_html=html_path, open_browser=False)

        # Save the JSON
        with open(json_path, "w") as f:
            json.dump({
                # "cell_connections": cell_connections,
                "cell_locations": cell_locations,
                **features,
                }, f, indent=2)
        print(f"[INFO] Saved design data to: {json_path}")

    return {
        "cell_locations": cell_locations,
        # "cell_connections": cell_connections,
        "validation_results": validation_results,
        "all_validations_passed": all_valid,
        **features,
    }


def run_batch_experiments(num_prompts=10, random_seed=2026, cell_limit=16, output_excel=None):
    """Run batch experiments with random prompts and save results to Excel."""

    # Initialize model at the start of batch experiments
    print(f"[INFO] Initializing model: {MODEL_KEY}")
    initialize_model(MODEL_KEY)

    # Generate output filename if not provided
    if output_excel is None:
        output_excel = f"CL{cell_limit}_{MODEL_KEY}_results_{DATA_NAME}.xlsx"

    print(f"[INFO] Starting batch experiments with {num_prompts} prompts (seed={random_seed})")
    print(f"[INFO] Using model: {MODEL_KEY}")
    print(f"[INFO] Using data: {DATA_NAME}")
    print(f"[INFO] Results will be saved to: {output_excel}")

    results = []

    for i in range(num_prompts):
        print(f"\n{'='*80}")
        print(f"Experiment {i+1}/{num_prompts}")
        print(f"{'='*80}")

        # Generate random prompt with seed
        prompt, req_voltage, req_capacity, req_width, req_depth, req_height = generate_random_prompt(random_seed + i, cell_limit=cell_limit)
        # prompt, req_voltage, req_width, req_depth, req_height, application = generate_random_prompt(random_seed + i)
        print(f"[INFO] Prompt: {prompt}")

        try:
            # Prepare required specs for validation
            required_specs = {
                "voltage": req_voltage,
                "capacity": req_capacity,
                "width_mm": req_width,
                "depth_mm": req_depth,
                "height_mm": req_height,
            }

            # Generate design
            design_result = main(prompt, required_specs=required_specs, render=False)

            if design_result is None:
                # Failed to generate valid design
                results.append({
                    "experiment_id": i + 1,
                    "prompt": prompt,
                    "required_voltage": req_voltage,
                    "required_capacity": req_capacity,
                    "required_width_mm": req_width,
                    "required_depth_mm": req_depth,
                    "required_height_mm": req_height,
                    # "application": application,
                    "generated_voltage": None,
                    "generated_capacity": None,
                    "generated_width_mm": None,
                    "generated_depth_mm": None,
                    "generated_height_mm": None,
                    "cell_locations": None,
                    # "cell_connections": None,
                    "series_count": None,
                    "parallel_count": None,
                    # "pybamm_validated": False,
                    "all_validations_passed": False,
                    "generation_status": "Failed"
                })
            else:
                # Successfully generated design
                validation_results = design_result.get("validation_results", {})
                results.append({
                    "experiment_id": i + 1,
                    "prompt": prompt,
                    "required_voltage": req_voltage,
                    "required_capacity": req_capacity,
                    "required_width_mm": req_width,
                    "required_depth_mm": req_depth,
                    "required_height_mm": req_height,
                    # "application": application,
                    "generated_voltage": design_result["generated_voltage"],
                    "generated_capacity": design_result["generated_capacity"],
                    "generated_width_mm": design_result["generated_width_mm"],
                    "generated_depth_mm": design_result["generated_depth_mm"],
                    "generated_height_mm": design_result["generated_height_mm"],
                    "cell_locations": str(design_result["cell_locations"]),
                    # "cell_connections": str(design_result["cell_connections"]),
                    "series_count": design_result["series_count"],
                    "parallel_count": design_result["parallel_count"],
                    # "pybamm_validated": design_result["pybamm_result"],
                    "all_validations_passed": design_result.get("all_validations_passed", False),
                    "generation_status": "Success"
                })

        except torch.cuda.OutOfMemoryError as e:
            print(f"\n{'='*80}")
            print(f"[FATAL] CUDA OUT OF MEMORY during experiment {i+1}")
            print(f"[FATAL] Error: {e}")
            print(f"[FATAL] Terminating batch experiment...")
            print(f"{'='*80}")

            # Save partial results before terminating
            if results:
                df = pd.DataFrame(results)
                os.makedirs("./model/RAG/output", exist_ok=True)
                partial_excel = f"PARTIAL_{output_excel}"
                excel_path = os.path.join("./model/RAG/output", partial_excel)
                df.to_excel(excel_path, index=False, engine='openpyxl')
                print(f"[INFO] Partial results ({len(results)} experiments) saved to: {excel_path}")

            sys.exit(1)

        except RuntimeError as e:
            # Catch CUDA OOM that may be raised as RuntimeError in some PyTorch versions
            if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
                print(f"\n{'='*80}")
                print(f"[FATAL] CUDA OUT OF MEMORY during experiment {i+1}")
                print(f"[FATAL] Error: {e}")
                print(f"[FATAL] Terminating batch experiment...")
                print(f"{'='*80}")

                # Save partial results before terminating
                if results:
                    df = pd.DataFrame(results)
                    os.makedirs("./model/RAG/output", exist_ok=True)
                    partial_excel = f"PARTIAL_{output_excel}"
                    excel_path = os.path.join("./model/RAG/output", partial_excel)
                    df.to_excel(excel_path, index=False, engine='openpyxl')
                    print(f"[INFO] Partial results ({len(results)} experiments) saved to: {excel_path}")

                sys.exit(1)
            else:
                # Re-raise other RuntimeErrors or handle as generic exception
                print(f"[ERROR] RuntimeError during experiment {i+1}: {e}")
                results.append({
                    "experiment_id": i + 1,
                    "prompt": prompt,
                    "required_voltage": req_voltage,
                    "required_capacity": req_capacity,
                    "required_width_mm": req_width,
                    "required_depth_mm": req_depth,
                    "required_height_mm": req_height,
                    # "application": application,
                    "generated_voltage": None,
                    "generated_capacity": None,
                    "generated_width_mm": None,
                    "generated_depth_mm": None,
                    "generated_height_mm": None,
                    "cell_locations": None,
                    # "cell_connections": None,
                    "series_count": None,
                    "parallel_count": None,
                    # "pybamm_validated": False,
                    "all_validations_passed": False,
                    "generation_status": f"Error: {str(e)}"
                })

        except Exception as e:
            print(f"[ERROR] Exception during experiment {i+1}: {e}")
            results.append({
                "experiment_id": i + 1,
                "prompt": prompt,
                "required_voltage": req_voltage,
                "required_capacity": req_capacity,
                "required_width_mm": req_width,
                "required_depth_mm": req_depth,
                "required_height_mm": req_height,
                # "application": application,
                "generated_voltage": None,
                "generated_capacity": None,
                "generated_width_mm": None,
                "generated_depth_mm": None,
                "generated_height_mm": None,
                "cell_locations": None,
                # "cell_connections": None,
                "series_count": None,
                "parallel_count": None,
                # "pybamm_validated": False,
                "all_validations_passed": False,
                "generation_status": f"Error: {str(e)}"
            })

    # Create DataFrame and save to Excel
    df = pd.DataFrame(results)

    # Save to output directory
    os.makedirs("./model/RAG/output", exist_ok=True)
    excel_path = os.path.join("./model/RAG/output", output_excel)
    df.to_excel(excel_path, index=False, engine='openpyxl')

    print(f"\n{'='*50}")
    print(f"[INFO] Batch experiments complete!")
    print(f"[INFO] Results saved to: {excel_path}")
    # print(f"[INFO] Success rate: {df['pybamm_validated'].sum()}/{num_prompts} ({df['pybamm_validated'].sum()/num_prompts*100:.1f}%)")
    print(f"[INFO] Success rate: {df['all_validations_passed'].sum()}/{num_prompts} ({df['all_validations_passed'].sum()/num_prompts*100:.1f}%)")
    print(f"{'='*50}")

    return df


if __name__ == "__main__":
    # Run batch experiments with random prompts
    run_batch_experiments(
        num_prompts=100,
        random_seed=2026, 
        cell_limit=16
    )

    # Single query example (commented out)
    # query = "I need a 14.8V and 10Ah battery pack under 120mm × 60mm × 40mm, using 18650 cells."
    # main(query, render=True)
