import ast
import re
import math
import numpy as np
from typing import Dict, List, Tuple, Optional
import pybamm

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

cell_radius = CELL_SPECS["diameter"] / 2.0
cell_spacing_h = CELL_SPECS["diameter"] + MIN_SPACING # Horizontal spacing between cells in a plane
cell_spacing_v = CELL_SPECS["length"] + MIN_SPACING # Vertical spacing between layers

# Hexagonal packing geometry in a plane
hex_x_spacing = cell_spacing_h  # Vertical distance between columns
hex_y_spacing = cell_spacing_h * (math.sqrt(3) / 2.0)  # Vertical distance between rows
hex_offset_x = cell_spacing_h / 2.0  # Horizontal offset for odd rows


class ValidationResult:
    """Store validation results with detailed error messages."""

    def __init__(self):
        self.is_valid = True
        self.errors = []
        self.warnings = []
        self.details = {}

    def add_error(self, message: str):
        """Add an error message and mark validation as failed."""
        self.is_valid = False
        self.errors.append(message)

    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)

    def add_detail(self, key: str, value):
        """Add detail information."""
        self.details[key] = value

    def __str__(self):
        """Return a formatted string representation."""
        lines = []
        lines.append(f"Validation {'PASSED' if self.is_valid else 'FAILED'}")

        if self.errors:
            lines.append("\nErrors:")
            for error in self.errors:
                lines.append(f"  ❌ {error}")

        if self.warnings:
            lines.append("\nWarnings:")
            for warning in self.warnings:
                lines.append(f"  ⚠️  {warning}")

        if self.details:
            lines.append("\nDetails:")
            for key, value in self.details.items():
                lines.append(f"  • {key}: {value}")

        return "\n".join(lines)


### Extract Design Information from Model Response

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
        "generated_voltage": extract_scalar("design_voltage", text),
        "generated_capacity": extract_scalar("design_capacity", text),
        "generated_width_mm": extract_scalar("design_width", text),
        "generated_depth_mm": extract_scalar("design_depth", text),
        "generated_height_mm": extract_scalar("design_height", text),
        "series_count": extract_scalar("series_count", text),
        "parallel_count": extract_scalar("parallel_count", text),
    }


### Validations

def extract_grid_dimensions(cell_locations: List[List[int]]) -> Dict[str, int]:
    """
    Extract grid dimensions from cell locations.

    Args:
        cell_locations: List of [x, y, z] or [x, y, z, present] coordinates

    Returns:
        Dict with keys: num_cells_width, num_cells_depth, num_cells_height
    """
    if not cell_locations:
        return {"num_cells_width": 0, "num_cells_depth": 0, "num_cells_height": 0}

    # Filter for present cells (if 4th element exists, check if it's 1)
    active_cells = []
    for loc in cell_locations:
        if len(loc) >= 4:
            if loc[3] == 1:  # present flag
                active_cells.append(loc[:3])
        else:
            active_cells.append(loc[:3])

    if not active_cells:
        return {"num_cells_width": 0, "num_cells_depth": 0, "num_cells_height": 0}

    cells = np.array(active_cells)

    # Find max values (grid) in each dimension (add 1 because coordinates are 0-indexed)
    num_cells_width = int(np.max(cells[:, 0])) + 1
    num_cells_depth = int(np.max(cells[:, 1])) + 1
    num_cells_height = int(np.max(cells[:, 2])) + 1

    # Find max values (physical mm) in each dimension
    max_mm_width = (num_cells_width - 1) * hex_x_spacing + CELL_SPECS["diameter"] + 2 * SAFETY_MARGIN # adjust as the last cell has no MIN_SPACING
    max_mm_depth = (num_cells_depth - 1) * hex_y_spacing + CELL_SPECS["diameter"] + 2 * SAFETY_MARGIN + hex_offset_x # adjust as the last cell has no MIN_SPACING
    max_mm_height = (num_cells_height - 1) * cell_spacing_v + CELL_SPECS["length"] + 2 * SAFETY_MARGIN


    return {
        "num_cells_width": num_cells_width,
        "num_cells_depth": num_cells_depth,
        "num_cells_height": num_cells_height, 
        "max_mm_width": max_mm_width, 
        "max_mm_depth": max_mm_depth, 
        "max_mm_height": max_mm_height
    }


def validate_with_pybamm(features):
    try:
        model = pybamm.lithium_ion.SPM()
        parameter_values = model.default_parameter_values
        
        parameter_values.update({
            "Nominal cell capacity [A.h]": features["generated_capacity"],
            "Number of cells connected in series to make a battery": features["series_count"],
            "Number of electrodes connected in parallel to make a cell": features["parallel_count"],
        })
        
        sim = pybamm.Simulation(model, parameter_values=parameter_values)
        sim.solve([0, 600])
        return True
    
    except Exception as e:
        print(f"❌ PyBaMM validation failed: {e}")
        return False
    

def validate_design_dimensions(
        cell_locations: List[List[int]], 
        claimed_dimensions: Dict[str, float]
        )->ValidationResult:
    """
    Validate that battery pack fit within claimed dimensions.
    """
     
    result = ValidationResult()

    if not cell_locations:
        result.add_error("Cell locations list is empty")
        return result
    
    pack_dimentions = extract_grid_dimensions(cell_locations)

    actual_width = pack_dimentions["max_mm_width"]
    actual_depth = pack_dimentions["max_mm_depth"]
    actual_height = pack_dimentions["max_mm_height"]

    result.add_detail("calculated_width_mm", f"{actual_width:.1f}")
    result.add_detail("calculated_depth_mm", f"{actual_depth:.1f}")
    result.add_detail("calculated_height_mm", f"{actual_height:.1f}")

    # Compare with claimed dimensions
    claimed_width = claimed_dimensions.get("width_mm")
    claimed_depth = claimed_dimensions.get("depth_mm")
    claimed_height = claimed_dimensions.get("height_mm")

    tolerance_mm = 1.0  # Allow 1mm tolerance for rounding

    if claimed_width is not None:
        if actual_width > claimed_width + tolerance_mm:
            result.add_error(
                f"Actual width ({actual_width:.1f}mm) exceeds claimed width ({claimed_width}mm)"
            )
        elif actual_width < claimed_width - 10:  # More than 10mm smaller
            result.add_error(
                f"Actual width ({actual_width:.1f}mm) is much smaller than claimed ({claimed_width}mm)"
            )

    if claimed_depth is not None:
        if actual_depth > claimed_depth + tolerance_mm:
            result.add_error(
                f"Actual depth ({actual_depth:.1f}mm) exceeds claimed depth ({claimed_depth}mm)"
            )
        elif actual_depth < claimed_depth - 10:
            result.add_error(
                f"Actual depth ({actual_depth:.1f}mm) is much smaller than claimed ({claimed_depth}mm)"
            )

    if claimed_height is not None:
        if actual_height > claimed_height + tolerance_mm:
            result.add_error(
                f"Actual height ({actual_height:.1f}mm) exceeds claimed height ({claimed_height}mm)"
            )
        elif actual_height < claimed_height - 10:
            result.add_error(
                f"Actual height ({actual_height:.1f}mm) is much smaller than claimed ({claimed_height}mm)"
            )

    return result


def validate_cell_locations(
    cell_locations: List[List[int]]
    ) -> ValidationResult:
    """
    Validate that cell locations are physically valid and there are no duplicate cells. 
    """

    result = ValidationResult()

    if not cell_locations:
        result.add_error("Cell locations list is empty")
        return result

    # Filter for present cells (if 4th element exists, check if it's 1)
    active_cells = []
    for loc in cell_locations:
        if len(loc) >= 4:
            if loc[3] == 1:  # present flag
                active_cells.append(loc[:3])
        else:
            active_cells.append(loc[:3])

    if not active_cells:
        result.add_error("No active cells found in cell_locations")
        return result

    result.add_detail("active_cells_count", len(active_cells))

    # Convert to numpy array for easier processing
    cells = np.array(active_cells)

    # Check for duplicate cells (same x, y, z coordinates)
    unique_cells = np.unique(cells, axis=0)
    if len(unique_cells) < len(cells):
        duplicates = len(cells) - len(unique_cells)
        result.add_error(f"Found {duplicates} duplicate cell location(s)")

    return result


def validate_cell_count(
    cell_locations: List[List[int]],
    claimed_series: int,
    claimed_parallel: int
) -> ValidationResult:
    """
    Validate that actual cell count matches claimed series/parallel configuration.
    """

    result = ValidationResult()

    # Count actual cells
    actual_count = 0
    for loc in cell_locations:
        if len(loc) >= 4:
            if loc[3] == 1:  # present flag
                actual_count += 1
        else:
            actual_count += 1

    result.add_detail("actual_cell_count", actual_count)

    # Expected count from series/parallel
    expected_count = claimed_series * claimed_parallel
    result.add_detail("expected_count_from_sp", f"{claimed_series} × {claimed_parallel} = {expected_count}")

    if actual_count != expected_count:
        result.add_error(
            f"Cell count mismatch: found {actual_count} cells, but design claimed {claimed_series}S{claimed_parallel}P "
            f"claimed configuration requires {expected_count} cells"
        )
    else:
        result.add_detail("cell_count_check", f"✓ {actual_count} cells matches {claimed_series}S{claimed_parallel}P")

    return result


def validate_electrical_specs(
    claimed_series: int,
    claimed_parallel: int,
    claimed_voltage: float,
    claimed_capacity: float,
    voltage_tolerance: float = 0.1,
    capacity_tolerance: float = 0.1
) -> ValidationResult:
    """
    Validate that claimed electrical specs match the series/parallel configuration.
    """

    result = ValidationResult()

    # Calculate expected voltage (series affects voltage)
    expected_voltage = claimed_series * CELL_SPECS["nominal_voltage"]
    voltage_diff = abs(claimed_voltage - expected_voltage)

    result.add_detail("expected_voltage", f"{claimed_series} × {CELL_SPECS["nominal_voltage"]}V = {expected_voltage}V")
    result.add_detail("claimed_voltage", f"{claimed_voltage}V")

    if voltage_diff > voltage_tolerance:
        result.add_error(
            f"Voltage mismatch: {claimed_series}S should give {expected_voltage}V, "
            f"but claimed {claimed_voltage}V (difference: {voltage_diff:.2f}V)"
        )
    else:
        result.add_detail("voltage_calc_check", f"✓ {claimed_voltage}V matches {claimed_series}S configuration")

    # Calculate expected capacity (parallel affects capacity)
    expected_capacity = claimed_parallel * CELL_SPECS["nominal_capacity"]
    capacity_diff = abs(claimed_capacity - expected_capacity)

    result.add_detail("expected_capacity", f"{claimed_parallel} × {CELL_SPECS["nominal_capacity"]}Ah = {expected_capacity}Ah")
    result.add_detail("claimed_capacity", f"{claimed_capacity}Ah")

    if capacity_diff > capacity_tolerance:
        result.add_error(
            f"Capacity mismatch: {claimed_parallel}P should give {expected_capacity}Ah, "
            f"but claimed {claimed_capacity}Ah (difference: {capacity_diff:.2f}Ah)"
        )
    else:
        result.add_detail("capacity_calc_check", f"✓ {claimed_capacity}Ah matches {claimed_parallel}P configuration")

    return result


def validate_prompt_satisfaction(
    required_specs: Dict[str, float],
    generated_specs: Dict[str, float],
    tolerance: float = 0.1
    ) -> ValidationResult:
    """
    Validate that generated design satisfies the prompt requirements.
    """

    result = ValidationResult()

    # Check voltage - should be exact match (within tolerance)
    req_voltage = required_specs.get("voltage")
    gen_voltage = generated_specs.get("voltage")

    if req_voltage is not None and gen_voltage is not None:
        voltage_diff = abs(gen_voltage - req_voltage)
        if voltage_diff > tolerance:
            result.add_error(
                f"Voltage mismatch: required {req_voltage}V, generated {gen_voltage}V "
                f"(difference: {voltage_diff:.2f}V, tolerance: {tolerance}V)"
            )
        else:
            result.add_detail("voltage_check", f"✓ {gen_voltage}V matches {req_voltage}V")
    else:
        result.add_error("Voltage values missing for comparison")

    # Check capacity - generated should be >= required
    req_capacity = required_specs.get("capacity")
    gen_capacity = generated_specs.get("capacity")

    if req_capacity is not None and gen_capacity is not None:
        if gen_capacity < req_capacity:
            result.add_error(
                f"Capacity insufficient: required >={req_capacity}Ah, generated {gen_capacity}Ah "
                f"(shortfall: {req_capacity - gen_capacity:.2f}Ah)"
            )
        else:
            result.add_detail("capacity_check", f"✓ {gen_capacity}Ah >= {req_capacity}Ah")
    else:
        result.add_error("Capacity values missing for comparison")

    # Check dimensions - generated should be <= required
    dimensions = [
        ("width_mm", "Width"),
        ("depth_mm", "Depth"),
        ("height_mm", "Height")
    ]

    for dim_key, dim_name in dimensions:
        req_dim = required_specs.get(dim_key)
        gen_dim = generated_specs.get(dim_key)

        if req_dim is not None and gen_dim is not None:
            if gen_dim > req_dim:
                result.add_error(
                    f"{dim_name} exceeds limit: required <={req_dim}mm, generated {gen_dim}mm "
                    f"(excess: {gen_dim - req_dim:.2f}mm)"
                )
            else:
                result.add_detail(f"{dim_key}_check", f"✓ {gen_dim}mm <= {req_dim}mm")
        else:
            result.add_error(f"{dim_name} values missing for comparison")

    return result


# Validate design

def validate_design(
    output: str, 
    required_specs: Optional[Dict] = None
) -> Dict[str, ValidationResult]:
    """
    Comprehensive validation of a battery pack design.
    """

    results = {}

    cell_locations = extract_cell_locations(output)
    features = extract_design_features(output)

    # Check if extraction was successful
    if not cell_locations:
        error_result = ValidationResult()
        error_result.add_error("Failed to extract cell_locations from output")
        results["extraction_error"] = error_result
        return results

    if not all(features.values()):
        error_result = ValidationResult()
        missing_fields = [k for k, v in features.items() if v is None]
        error_result.add_error(f"Failed to extract required features: {', '.join(missing_fields)}")
        results["extraction_error"] = error_result
        return results

    # 1. validate_design_dimensions
    results["physical_dimension_validity"] = validate_design_dimensions(
        cell_locations=cell_locations,
        claimed_dimensions={
            "width_mm": float(features["generated_width_mm"]),
            "depth_mm": float(features["generated_depth_mm"]),
            "height_mm": float(features["generated_height_mm"]),
        }
    )

    # 2. validate_cell_locations
    results["cell_location_validity"] = validate_cell_locations(
        cell_locations=cell_locations
    )

    # 3. validate_cell_count
    results["cell_count_validity"] = validate_cell_count(
        cell_locations=cell_locations,
        claimed_series=int(features["series_count"]),
        claimed_parallel=int(features["parallel_count"])
    )

    # 4. validate_electrical_specs
    results["electrical_specs_validity"] = validate_electrical_specs(
        claimed_series=int(features["series_count"]),
        claimed_parallel=int(features["parallel_count"]),
        claimed_voltage=float(features["generated_voltage"]),
        claimed_capacity=float(features["generated_capacity"])
    )

    # 5. validate_prompt_satisfaction
    if required_specs:
        results["prompt_satisfaction_validity"] = validate_prompt_satisfaction(
            required_specs=required_specs,
            generated_specs={
                "voltage": float(features["generated_voltage"]),
                "capacity": float(features["generated_capacity"]),
                "width_mm": float(features["generated_width_mm"]),
                "depth_mm": float(features["generated_depth_mm"]),
                "height_mm": float(features["generated_height_mm"]),
            }
        )

    return results


def print_validation_summary(results: Dict[str, ValidationResult]) -> bool:
    """
    Print a comprehensive summary of all validation results.
    """
    
    print("\n" + "="*50)
    print("VALIDATION SUMMARY")

    all_valid = all(result.is_valid for result in results.values())

    for check_name, result in results.items():
        print(f"\n[{check_name.upper().replace('_', ' ')}]")
        print(result)

    print("\n" + "="*50)
    if all_valid:
        print("✅ ALL VALIDATIONS PASSED")
        return True
    else:
        print("❌ VALIDATION FAILED")
        failed_checks = [name for name, result in results.items() if not result.is_valid]
        print(f"Failed checks: {', '.join(failed_checks)}")
        return False


