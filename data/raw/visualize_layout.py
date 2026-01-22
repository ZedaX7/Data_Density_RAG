"""
Simple text-based visualization of hexagonal cell layout
"""
import math

CELL_SPECS = {
    "diameter": 18.0,
    "length": 65.0,
}

MIN_SPACING = 2.0
SAFETY_MARGIN = 5.0

def generate_hexagonal_layout(num_cells_w, num_cells_d, num_cells_h):
    """Generate hexagonal close-packed cell layout."""
    cell_positions = []
    cell_grid_coords = []

    cell_radius = CELL_SPECS["diameter"] / 2.0
    cell_spacing = cell_radius * 2 + MIN_SPACING

    hex_offset_x = cell_spacing * math.sqrt(3) / 2.0
    hex_offset_y = cell_spacing * 0.5

    for z in range(num_cells_h):
        z_pos = SAFETY_MARGIN + z * (CELL_SPECS["length"] + MIN_SPACING)

        for y in range(num_cells_d):
            for x in range(num_cells_w):
                if y % 2 == 1:
                    x_pos = SAFETY_MARGIN + x * hex_offset_x * 2 + hex_offset_x
                else:
                    x_pos = SAFETY_MARGIN + x * hex_offset_x * 2

                y_pos = SAFETY_MARGIN + y * (cell_spacing - hex_offset_y)

                cell_positions.append([round(x_pos, 2), round(y_pos, 2), round(z_pos, 2)])
                cell_grid_coords.append([x, y, z])

    return cell_positions, cell_grid_coords


def visualize_layer(cell_positions, cell_grid_coords, layer_z):
    """Visualize a single Z-layer of cells in text."""
    # Filter cells at this Z layer
    layer_cells = [(i, pos, grid) for i, (pos, grid) in enumerate(zip(cell_positions, cell_grid_coords))
                   if grid[2] == layer_z]

    if not layer_cells:
        print(f"No cells at layer Z={layer_z}")
        return

    print(f"\nLayer Z={layer_z} (Height: {layer_cells[0][1][2]:.1f}mm)")
    print("-" * 60)

    # Group by Y coordinate
    cells_by_row = {}
    for cell_id, pos, grid in layer_cells:
        y_grid = grid[1]
        if y_grid not in cells_by_row:
            cells_by_row[y_grid] = []
        cells_by_row[y_grid].append((cell_id, pos, grid))

    # Print each row
    for y_grid in sorted(cells_by_row.keys()):
        row_cells = sorted(cells_by_row[y_grid], key=lambda c: c[1][0])  # Sort by x position

        row_label = f"Row Y={y_grid}"
        if y_grid % 2 == 0:
            row_label += " (Even) "
        else:
            row_label += " (Odd)  "

        print(f"{row_label}: ", end="")

        # Show spacing for odd rows
        if y_grid % 2 == 1:
            print("  ", end="")  # Visual offset for odd rows

        for cell_id, pos, grid in row_cells:
            print(f"[{cell_id}]", end=" ")

        # Show actual x-coordinates
        print()
        print(" " * len(row_label) + "  ", end="")
        if y_grid % 2 == 1:
            print("  ", end="")
        for cell_id, pos, grid in row_cells:
            print(f"{pos[0]:4.0f}", end=" ")
        print()


def visualize_design(num_cells_w, num_cells_d, num_cells_h):
    """Visualize complete hexagonal design."""
    print("=" * 60)
    print(f"Hexagonal Layout Visualization: {num_cells_w}x{num_cells_d}x{num_cells_h}")
    print("=" * 60)

    cell_positions, cell_grid_coords = generate_hexagonal_layout(num_cells_w, num_cells_d, num_cells_h)

    print(f"\nTotal cells: {len(cell_positions)}")
    print(f"Expected: {num_cells_w * num_cells_d * num_cells_h}")

    # Calculate dimensions
    if cell_positions:
        max_x = max(p[0] for p in cell_positions) + CELL_SPECS["diameter"]/2 + SAFETY_MARGIN
        max_y = max(p[1] for p in cell_positions) + CELL_SPECS["diameter"]/2 + SAFETY_MARGIN
        max_z = max(p[2] for p in cell_positions) + CELL_SPECS["length"] + SAFETY_MARGIN

        print(f"\nPack dimensions: {max_x:.1f} x {max_y:.1f} x {max_z:.1f} mm")

    # Visualize each layer
    for z in range(num_cells_h):
        visualize_layer(cell_positions, cell_grid_coords, z)

    print("\n" + "=" * 60)
    print("Legend:")
    print("  [N] = Cell ID N")
    print("  Numbers below = X-coordinate in mm")
    print("  Even rows: No offset")
    print("  Odd rows:  Offset to the right (hexagonal packing)")
    print("=" * 60)


if __name__ == "__main__":
    # Example 1: Small 3x3x1 layout
    visualize_design(3, 3, 1)

    print("\n\n")

    # Example 2: 2S2P configuration (4 cells = 2x1x2)
    print("\n" + "=" * 60)
    print("Example: 2S2P Configuration (4 cells)")
    print("=" * 60)
    visualize_design(2, 1, 2)

    print("\n\n")

    # Example 3: 3S2P configuration (6 cells = 2x1x3)
    print("\n" + "=" * 60)
    print("Example: 3S2P Configuration (6 cells)")
    print("=" * 60)
    visualize_design(2, 1, 3)
