import numpy as np
import plotly.graph_objects as go

CLASS_COLORS = {
    "wall": "rgb(50, 50, 200)",
    "door": "rgb(50, 200, 50)",
    "window": "rgb(50, 200, 200)",
    "ceiling": "rgb(200, 200, 200)",
    "floor": "rgb(100, 100, 100)",
    "boiler": "rgb(255, 140, 0)",
    "pipe": "rgb(200, 50, 50)",
    "valve": "rgb(50, 200, 50)"
}

def get_class_color(class_name):
    key = class_name.lower().strip()
    if key in CLASS_COLORS:
        return CLASS_COLORS[key]
    
    hash_val = sum(ord(c) for c in key)
    r = (hash_val * 50) % 255
    g = (hash_val * 80) % 255
    b = (hash_val * 110) % 255
    return f"rgb({r},{g},{b})"

def get_box_wireframe(box):
    center = np.array(box['center'])
    scale = np.array(box['scale'])
    rotation = np.array(box['rotation'])
    
    corners = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
    ])
    
    transformed_corners = (corners * scale) @ rotation + center
    
    lines_indices = [
        0, 1, 1, 2, 2, 3, 3, 0,
        4, 5, 5, 6, 6, 7, 7, 4,
        0, 4, 1, 5, 2, 6, 3, 7
    ]
    
    x_lines, y_lines, z_lines = [], [], []
    
    for i in range(0, len(lines_indices), 2):
        start = transformed_corners[lines_indices[i]]
        end = transformed_corners[lines_indices[i+1]]
        x_lines.extend([start[0], end[0], None])
        y_lines.extend([start[1], end[1], None])
        z_lines.extend([start[2], end[2], None])
        
    return x_lines, y_lines, z_lines

def create_plot(points, colors, boxes=None, point_size=1.5, max_points=20000):
    data_traces = []
    
    # Track presence of points to handle toggle logic correctly
    has_points = False

    # 1. Plot Point Cloud
    if points is not None and max_points > 0:
        has_points = True
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            pts_sub = points[indices]
            col_sub = colors[indices]
        else:
            pts_sub = points
            col_sub = colors

        if col_sub.max() > 1.1:
            col_sub_255 = col_sub.astype(int)
        else:
            col_sub_255 = (col_sub * 255).astype(int)
        
        color_strings = [f'rgb({r},{g},{b})' for r, g, b in col_sub_255]

        data_traces.append(go.Scatter3d(
            x=pts_sub[:, 0],
            y=pts_sub[:, 1],
            z=pts_sub[:, 2],
            mode='markers',
            marker=dict(size=point_size, color=color_strings, opacity=0.8),
            name='Point Cloud',
            hoverinfo='none'
        ))

    # 2. Plot Boxes
    if boxes:
        boxes_by_class = {}
        for box in boxes:
            cls = box.get('label', 'object')
            if cls not in boxes_by_class:
                boxes_by_class[cls] = []
            boxes_by_class[cls].append(box)
        
        for cls, class_boxes in boxes_by_class.items():
            cls_color = get_class_color(cls)
            x_lines, y_lines, z_lines = [], [], []
            text_x, text_y, text_z, text_labels = [], [], [], []

            for box in class_boxes:
                # Wireframe
                bx, by, bz = get_box_wireframe(box)
                x_lines.extend(bx)
                y_lines.extend(by)
                z_lines.extend(bz)
                
                # Label
                center = box['center']
                text_x.append(center[0])
                text_y.append(center[1])
                text_z.append(center[2])
                text_labels.append(cls)

            # Lines Trace
            data_traces.append(go.Scatter3d(
                x=x_lines, y=y_lines, z=z_lines,
                mode='lines',
                line=dict(color=cls_color, width=4),
                name=f"Box: {cls}",
                legendgroup=cls,
                showlegend=True
            ))
            
            # Text Trace
            data_traces.append(go.Scatter3d(
                x=text_x, y=text_y, z=text_z,
                mode='text',
                text=text_labels,
                textposition="middle center",
                textfont=dict(size=10, color=cls_color),
                hoverinfo='skip',
                legendgroup=cls,
                showlegend=False
            ))

    fig = go.Figure(data=data_traces)
    
    # --- CLIENT-SIDE TOGGLE BUTTON (Moved to Bottom-Left) ---
    n_traces = len(data_traces)
    
    if n_traces > 0:
        vis_all = [True] * n_traces
        
        if has_points:
            vis_none = [True] + [False] * (n_traces - 1)
        else:
            vis_none = [False] * n_traces

        updatemenus = [
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"visible": vis_all}],
                        label="Show Boxes",
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": vis_none}],
                        label="Hide Boxes",
                        method="restyle"
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                # --- POSITIONING CHANGES START ---
                x=0.01,
                xanchor="left",
                y=0.01,            # Bottom
                yanchor="bottom",  # Anchored to bottom edge
                # --- POSITIONING CHANGES END ---
                bgcolor="rgba(255, 255, 255, 0.8)"
            ),
        ]
        
        fig.update_layout(updatemenus=updatemenus)

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.05, bgcolor="rgba(255,255,255,0.5)")
    )
    return fig