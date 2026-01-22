import numpy as np
import plotly.graph_objects as go

def render_battery_pack(cell_locations, output_html="battery_pack_rendering.html", open_browser=False):
    """
    Render a 3D cylindrical battery pack with top and bottom caps using Plotly.

    Parameters:
        cell_locations (list of [x, y, z, present]):
            List of cell coordinates and presence flags.
        output_html (str):
            Output filename for the interactive HTML visualization.
        open_browser (bool):
            Whether to automatically open the HTML file in the browser.
    """
    RADIUS = 9   # mm
    HEIGHT = 65  # mm
    RESOLUTION = 30

    def create_cell_surfaces(x0, y0, z0):
        theta = np.linspace(0, 2 * np.pi, RESOLUTION)
        z = np.array([0, HEIGHT])
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_side = RADIUS * np.cos(theta_grid) + x0
        y_side = RADIUS * np.sin(theta_grid) + y0
        z_side = z_grid + z0

        r_circle = np.linspace(0, RADIUS, 2)
        r_grid, theta_grid = np.meshgrid(r_circle, theta)
        x_top = r_grid * np.cos(theta_grid) + x0
        y_top = r_grid * np.sin(theta_grid) + y0
        z_top = np.full_like(x_top, z0 + HEIGHT)
        z_bot = np.full_like(x_top, z0)

        return (x_side, y_side, z_side), (x_top, y_top, z_top), (x_top, y_top, z_bot)

    fig = go.Figure()

    for idx, (x, y, z, present) in enumerate(cell_locations):
        if present:
            cx = x * 2 * RADIUS
            cy = y * 2 * RADIUS
            cz = z * HEIGHT
            (xs, ys, zs), (xt, yt, zt), (xb, yb, zb) = create_cell_surfaces(cx, cy, cz)

            # Side
            fig.add_trace(go.Surface(x=xs, y=ys, z=zs, colorscale='Blues', showscale=False, hoverinfo='skip'))
            # Top
            fig.add_trace(go.Surface(x=xt, y=yt, z=zt, colorscale='Blues', showscale=False, hoverinfo='skip'))
            # Bottom
            fig.add_trace(go.Surface(x=xb, y=yb, z=zb, colorscale='Blues', showscale=False, hoverinfo='skip'))

    fig.update_layout(
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data'
        ),
        title="3D Battery Pack Visualization with 18650 Cells",
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False
    )

    fig.write_html(output_html, auto_open=open_browser)

