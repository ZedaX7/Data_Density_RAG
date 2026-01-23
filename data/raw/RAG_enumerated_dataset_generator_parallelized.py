from torch.utils.data import Dataset
import json
import math
import pybamm
import traceback
import numpy as np
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import os
import time

# 18650 Cell specifications
CELL_SPECS = {
    "nominal_voltage": 3.7,
    "max_voltage": 4.2,
    "min_voltage": 2.5,
    "nominal_capacity": 2.5,
    "weight": 0.045,
    "diameter": 18.0,
    "length": 65.0,
    "internal_resistance": 0.05,
    "max_discharge_rate": 10,
    "max_charge_rate": 2,
}

# Design constraints - USER CONFIGURABLE
MIN_SPACING = 2.0  # mm between cells (x)
SAFETY_MARGIN = 5.0  # mm between cells and pack wall (y)

# Maximum cells in each dimension - USER CONFIGURABLE
MAX_CELLS_WIDTH = 64  # Maximum number of cells in width
MAX_CELLS_DEPTH = 64  # Maximum number of cells in depth
MAX_CELLS_HEIGHT = 4  # Maximum number of cells in height (Z-layers)

# Maximum series and parallel configurations - USER CONFIGURABLE
MAX_SERIES = 64
MAX_PARALLEL = 64

# Parallelization settings - USER CONFIGURABLE
NUM_WORKERS = 16 #cpu_count()  # Use all available CPU cores, or set to specific number

dataset_name = f"[{MAX_CELLS_WIDTH}-{MAX_CELLS_DEPTH}-{MAX_CELLS_HEIGHT}-{MAX_SERIES}-{MAX_PARALLEL}]"
save_path = f"./data/raw/full_{dataset_name}/enumerated_battery_pack_dataset_{dataset_name}.json"

def generate_hexagonal_layout(num_cells_w, num_cells_d, num_cells_h):
    """
    Generate hexagonal close-packed cell layout with physical coordinates.

    Returns:
        cell_positions: List of [x_mm, y_mm, z_mm] for each cell
        cell_grid_coords: List of [grid_x, grid_y, grid_z] for each cell
    """
    cell_positions = []  # Physical coordinates in mm, using the center of gravity
    cell_grid_coords = []  # Grid coordinates for reference

    cell_radius = CELL_SPECS["diameter"] / 2.0
    cell_spacing_h = cell_radius * 2 + MIN_SPACING # Horizontal spacing between cells in a plane
    cell_spacing_v = CELL_SPECS["length"] + MIN_SPACING # Vertical spacing between layers

    # Hexagonal packing geometry in a plane
    hex_x_spacing = cell_spacing_h  # Vertical distance between columns
    hex_y_spacing = cell_spacing_h * (math.sqrt(3) / 2.0)  # Vertical distance between rows
    hex_offset_x = cell_spacing_h / 2.0  # Horizontal offset for odd rows

    cell_id = 0

    for z in range(num_cells_h):
        z_pos = SAFETY_MARGIN + CELL_SPECS["length"] / 2.0 + z * cell_spacing_v  # Safe margin + half length/height + # of spacing

        for y in range(num_cells_d):
            for x in range(num_cells_w):
                # Hexagonal packing: offset every other row
                if y % 2 == 1: # Odd row
                    x_pos = SAFETY_MARGIN + cell_radius + x * hex_x_spacing + hex_offset_x # Safe margin + radius + # of spacing + offset
                else: # Even row
                    x_pos = SAFETY_MARGIN + cell_radius + x * hex_x_spacing # Safe margin + radius + # of spacing

                y_pos = SAFETY_MARGIN + cell_radius + y * hex_y_spacing # Safe margin + radius + # of spacing * sqrt(3)/2

                cell_positions.append([round(x_pos, 2), round(y_pos, 2), round(z_pos, 2)])
                cell_grid_coords.append([x, y, z])
                cell_id += 1

    return cell_positions, cell_grid_coords


def calculate_pack_dimensions_from_cells(cell_positions):
    """
    Calculate actual pack dimensions based on placed cells.
    """
    if not cell_positions:
        return 0, 0, 0

    cell_positions_array = np.array(cell_positions) # Physical coordinates in mm

    max_x = np.max(cell_positions_array[:, 0]) + CELL_SPECS["diameter"]/2.0 + SAFETY_MARGIN # max + radius + margin
    max_y = np.max(cell_positions_array[:, 1]) + CELL_SPECS["diameter"]/2.0 + SAFETY_MARGIN # max + radius + margin
    max_z = np.max(cell_positions_array[:, 2]) + CELL_SPECS["length"]/2.0 + SAFETY_MARGIN # max + half length/height + margin

    return round(max_x, 1), round(max_y, 1), round(max_z, 1)


def generate_cell_locations(cell_positions, cell_grid_coords):
    """
    Generate cell location data with grid coordinates.
    Format: [grid_x, grid_y, grid_z]
    """
    cell_locations = []
    for grid in cell_grid_coords:
        cell_locations.append([
            grid[0],  # grid x
            grid[1],  # grid y
            grid[2],  # grid z
        ])
    return cell_locations


def validate_series_parallel_config(series_count, parallel_count, total_cells):
    """
    Validate that series and parallel configuration is feasible.
    - All parallel strings must have exactly the same number of series cells
    - Total cells must equal series_count * parallel_count
    """
    expected_cells = series_count * parallel_count
    if total_cells != expected_cells:
        return False
    return True

# def generate_series_parallel_connections(series_count, parallel_count, cell_positions):
#     """
#     Generate physically valid series-parallel connections.

#     Strategy:
#     1. Organize cells into parallel strings (each with 'series_count' cells)
#     2. Connect cells within each string in series (vertically, same x-y position)
#     3. Connect parallel strings at corresponding positions

#     Returns:
#         connections: List of [from_cell_id, to_cell_id, connection_type]
#                      connection_type: 1 = series, 0 = parallel
#     """
#     total_cells = series_count * parallel_count
#     connections = []

#     if total_cells <= 1:
#         return connections

#     # Group cells by (x, y) position for vertical series connections
#     cells_by_xy = {}
#     for cell_id, pos in enumerate(cell_positions[:total_cells]):
#         xy_key = (round(pos[0], 1), round(pos[1], 1))
#         if xy_key not in cells_by_xy:
#             cells_by_xy[xy_key] = []
#         cells_by_xy[xy_key].append((cell_id, pos[2]))  # (cell_id, z_position)

#     # Sort cells in each position by z-coordinate (bottom to top)
#     for xy_key in cells_by_xy:
#         cells_by_xy[xy_key].sort(key=lambda x: x[1])

#     # Create series connections within each vertical stack
#     parallel_strings = []
#     for xy_key, cells in cells_by_xy.items():
#         if len(cells) >= series_count:
#             # Take first 'series_count' cells for this string
#             string_cells = [cell[0] for cell in cells[:series_count]]
#             parallel_strings.append(string_cells)

#             # Create series connections within this string
#             for i in range(len(string_cells) - 1):
#                 connections.append([string_cells[i], string_cells[i+1], 1])

#     # Verify we have the right number of parallel strings
#     if len(parallel_strings) != parallel_count:
#         # Fallback: linear arrangement if spatial grouping fails
#         parallel_strings = []
#         for p in range(parallel_count):
#             start_idx = p * series_count
#             string = list(range(start_idx, start_idx + series_count))
#             parallel_strings.append(string)

#             # Series connections within string
#             for i in range(series_count - 1):
#                 connections.append([string[i], string[i+1], 1])

#     # Create parallel connections between strings at corresponding positions
#     if parallel_count > 1:
#         for pos_in_string in range(series_count):
#             for p in range(parallel_count - 1):
#                 cell1 = parallel_strings[p][pos_in_string]
#                 cell2 = parallel_strings[p + 1][pos_in_string]
#                 connections.append([cell1, cell2, 0])

#     return connections


def validate_with_pybamm(features):
    """
    Validate battery design using PyBaMM simulation.
    Suppresses print output for parallel execution.
    """
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

    except Exception:
        # Suppress error output in parallel mode
        # print(f"❌ PyBaMM validation failed: {e}")
        # traceback.print_exc()
        return False


def process_series_parallel_config(series_parallel_tuple):
    """
    Process a single (series, parallel) configuration.
    This function is called in parallel by worker processes.

    Returns:
        List of valid samples for this configuration
    """
    series, parallel = series_parallel_tuple
    total_cells = series * parallel
    local_samples = []

    # Enumerate all grid dimensions that can fit the required cells
    for num_cells_w in range(1, MAX_CELLS_WIDTH + 1):
        for num_cells_d in range(1, MAX_CELLS_DEPTH + 1):
            cells_per_layer = num_cells_w * num_cells_d

            # Calculate required height layers
            if cells_per_layer == 0:
                continue

            min_h = math.ceil(total_cells / cells_per_layer)
            max_h = min(MAX_CELLS_HEIGHT, min_h + 2)  # Allow some flexibility

            for num_cells_h in range(min_h, max_h + 1):
                grid_capacity = num_cells_w * num_cells_d * num_cells_h

                # Only generate if grid can exactly fit the cells
                if grid_capacity != total_cells:
                    continue

                # Validate configuration
                if not validate_series_parallel_config(series, parallel, total_cells):
                    continue

                # Generate hexagonal cell layout
                cell_positions, cell_grid_coords = generate_hexagonal_layout(
                    num_cells_w, num_cells_d, num_cells_h
                )

                # Calculate pack dimensions
                width_mm, depth_mm, height_mm = calculate_pack_dimensions_from_cells(cell_positions)

                # Calculate pack properties
                pack_voltage = series * CELL_SPECS["nominal_voltage"]
                pack_capacity = parallel * CELL_SPECS["nominal_capacity"]
                pack_weight = total_cells * CELL_SPECS["weight"]
                pack_energy = pack_voltage * pack_capacity

                features = {
                    "num_cells_width": num_cells_w,
                    "num_cells_depth": num_cells_d,
                    "num_cells_height": num_cells_h,
                    "total_cells": total_cells,
                    "series_count": series,
                    "parallel_count": parallel,
                    "weight_kg": round(pack_weight, 3),
                    "nominal_voltage": round(pack_voltage, 1),
                    "capacity_ah": round(pack_capacity, 1),
                    "energy_wh": round(pack_energy, 1),
                    "max_voltage": round(series * CELL_SPECS["max_voltage"], 1),
                    "min_voltage": round(series * CELL_SPECS["min_voltage"], 1),
                    "max_discharge_current": round(parallel * CELL_SPECS["nominal_capacity"] * CELL_SPECS["max_discharge_rate"], 1),
                    "max_charge_current": round(parallel * CELL_SPECS["nominal_capacity"] * CELL_SPECS["max_charge_rate"], 1),
                    "physical_width_mm": round(width_mm, 1),
                    "physical_depth_mm": round(depth_mm, 1),
                    "physical_height_mm": round(height_mm, 1),
                    "energy_density_wh_kg": round(pack_energy / pack_weight, 1),
                    "internal_resistance_ohm": round(series * CELL_SPECS["internal_resistance"] / parallel, 4),
                    "configuration_type": f"{series}S{parallel}P {num_cells_w}x{num_cells_d}x{num_cells_h}",
                }

                # PyBaMM validation
                if validate_with_pybamm(features):
                    # Generate cell locations
                    cell_locations = generate_cell_locations(cell_positions, cell_grid_coords)

                    sample = {
                        "features": features,
                        "cell_locations": cell_locations,
                        "pybamm_validated": True,
                        "generation_method": "exhaustive_hexagonal_enumeration_parallel"
                    }
                    local_samples.append(sample)

    return local_samples


class Synthetic_Battery_Pack_Dataset_Parallel(Dataset):
    def __init__(self, num_workers=None):
        self.data = []

        if num_workers is None:
            num_workers = NUM_WORKERS

        print(f"\n🔄 Exhaustively generating all feasible battery pack designs with hexagonal packing...")
        print(f"🚀 Using {num_workers} parallel workers (CPU cores: {cpu_count()})")
        print(f"Design constraints:")
        print(f"  - Max cells: W={MAX_CELLS_WIDTH}, D={MAX_CELLS_DEPTH}, H={MAX_CELLS_HEIGHT}")
        print(f"  - Max series: {MAX_SERIES}, Max parallel: {MAX_PARALLEL}")
        print(f"  - Cell spacing: {MIN_SPACING}mm, Wall margin: {SAFETY_MARGIN}mm")

        # Generate all (series, parallel) combinations to process
        series_parallel_configs = [
            (series, parallel)
            for series in range(1, MAX_SERIES + 1)
            for parallel in range(1, MAX_PARALLEL + 1)
        ]

        total_configs = len(series_parallel_configs)
        print(f"  - Total series/parallel configurations to process: {total_configs}")

        # Process configurations in parallel
        print(f"\n⚙️  Processing configurations in parallel...")
        start_time = time.time()

        with Pool(processes=num_workers) as pool:
            # Use imap_unordered for better progress tracking
            results = pool.imap_unordered(process_series_parallel_config, series_parallel_configs, chunksize=1)

            configs_processed = 0
            for result_samples in results:
                self.data.extend(result_samples)
                configs_processed += 1

                # Progress update every 100 configurations
                if configs_processed % 100 == 0:
                    elapsed_time = time.time() - start_time
                    print(f"  Processed {configs_processed}/{total_configs} configs, found {len(self.data)} valid designs... (Elapsed: {elapsed_time:.1f}s)")

        end_time = time.time()
        total_time = end_time - start_time
        print(f"\n✅ Generated {len(self.data)} valid designs from {total_configs} series/parallel configurations")
        print(f"⏱️  Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def export_to_json(dataset, filename):
    export_data = dataset.data
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(export_data, f, indent=2)
    print(f"\n💾 Dataset exported to {filename}")


if __name__ == "__main__":
    # Important for multiprocessing on Windows
    import multiprocessing
    multiprocessing.freeze_support()

    dataset = Synthetic_Battery_Pack_Dataset_Parallel()
    if len(dataset) > 0:
        export_to_json(dataset, save_path)
        print(f"🎉 All {len(dataset)} designs passed PyBaMM simulation")
    else:
        print("❌ No valid designs generated.")
