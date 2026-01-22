import ast
import re
import os
import json
from datetime import datetime
import pybamm

from retrieve import retrieve_relevant_docs
from generate import generate_answer
from visual_rendering import render_battery_pack

def validate_with_pybamm(features):
    try:
        model = pybamm.lithium_ion.SPM()
        parameter_values = model.default_parameter_values
        
        parameter_values.update({
            "Nominal cell capacity [A.h]": features["capacity_ah"],
            "Number of cells connected in series to make a battery": features["series_count"],
            "Number of electrodes connected in parallel to make a cell": features["parallel_count"],
        })
        
        sim = pybamm.Simulation(model, parameter_values=parameter_values)
        sim.solve([0, 600])
        return True
    
    except Exception as e:
        print(f"❌ PyBaMM validation failed: {e}")
        return False
    

def extract_list_block(field_name, text):
    """
    Extracts a bracketed list from the text for a given field name.
    Handles optional prefixes (-, *), any casing, and multiline brackets.
    """
    # Match field name with optional prefix and any casing
    pattern = rf"(?:[-*]?\s*)?{field_name}\s*[:=]\s*\["
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        print(f"[DEBUG] Start of {field_name} block not found.")
        return None

    # Walk forward from the matched '[' to find balanced brackets
    start = match.end() - 1
    bracket_count = 1
    end = start
    while end < len(text) - 1 and bracket_count > 0:
        end += 1
        if text[end] == "[":
            bracket_count += 1
        elif text[end] == "]":
            bracket_count -= 1

    try:
        raw_block = text[start:end + 1]
        return ast.literal_eval(raw_block)
    except Exception as e:
        print(f"[ERROR] Failed to parse {field_name}: {e}")
        return None


def extract_scalar(field_name, text):
    """Extract a numeric scalar value for field_name from text."""
    pattern = rf"{field_name}\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)"
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    print(f"[DEBUG] {field_name} not found or invalid.")
    return None


def extract_cell_locations(text):
    return extract_list_block("cell_locations", text)

def extract_cell_connections(text):
    return extract_list_block("cell_connections", text)

def extract_design_features(text):
    return {
        "capacity_ah": extract_scalar("capacity_ah", text),
        "series_count": extract_scalar("series_count", text),
        "parallel_count": extract_scalar("parallel_count", text),
    }


def generate_timestamped_basename(prefix="battery_pack_rendering"):
    """Generate a shared basename using timestamp for HTML and JSON files."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    return f"{prefix}_{timestamp}"


def main(query, render=False):
    docs = retrieve_relevant_docs(query)
    while True:
        output = generate_answer(query, docs)
        print(output)

        cell_locations = extract_cell_locations(output)
        cell_connections = extract_cell_connections(output)
        features = extract_design_features(output)

        # Check parsing success
        if not cell_locations or not all(features.values()):
            print("[WARNING] Incomplete design extracted; retrying generation...")
            continue

        # Validate with PyBaMM
        if validate_with_pybamm(features):
            print("✅ Design validated successfully.")
            break
        else:
            print("[WARNING] Design invalid; retrying generation...")
            continue

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
            "cell_connections": cell_connections,
            "cell_locations": cell_locations,
            **features
            }, f, indent=2)
    print(f"[INFO] Saved design data to: {json_path}")


if __name__ == "__main__":

    query = "I need a 14.8V battery pack under 120mm × 60mm × 40mm, optimized for high current drones, using 18650 cells."
    # query = "I need a 11.1V battery pack under 1000mm × 1000mm × 1000mm, optimized for high current drones, using 18650 cells."
    # query = "I need a 88.8V battery pack under 200mm × 200mm × 200mm, optimized for high current drones, using 18650 cells."
    main(query, render=True)

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Generate and validate a battery pack design.")
#     parser.add_argument("query", help="Prompt for design generation")
#     parser.add_argument("--render", action="store_true", help="Enable HTML rendering")
#     args = parser.parse_args()
#     main(args.query, render=args.render)
