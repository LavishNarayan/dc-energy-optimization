from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def build_rack_thermal_heatmap(
    t_in: np.ndarray,
    t_out: np.ndarray,
    cpu_util: np.ndarray,
    rack_it_kw: np.ndarray,
    safe_max_c: float = 27.0,
) -> go.Figure:
    num_racks = len(t_in)
    rack_labels = [f"Rack {i:02d}" for i in range(num_racks)]

    z_exhaust = np.clip((t_out - 22.0) / (42.0 - 22.0), 0.0, 1.0)
    z_inlet = np.clip((t_in - 16.0) / (28.5 - 16.0), 0.0, 1.0)
    z_util = np.clip(cpu_util, 0.0, 1.0)

    z_matrix = np.vstack([z_exhaust, z_inlet, z_util])

    text_matrix = [
        [f"{val:.1f}°C" for val in t_out],
        [
            f"{val:.1f}°C" + (" ⚠️" if val > safe_max_c else "")
            for val in t_in
        ],
        [f"{u * 100:.0f}% ({kw:.1f} kW)" for u, kw in zip(cpu_util, rack_it_kw)],
    ]

    hover_matrix = [
        [
            f"<b>{rack_labels[i]} — Hot Aisle Exhaust</b><br>"
            f"Exhaust Temp: {t_out[i]:.2f} °C<br>"
            f"Delta-T Across Rack: +{t_out[i] - t_in[i]:.2f} °C"
            for i in range(num_racks)
        ],
        [
            f"<b>{rack_labels[i]} — Cold Aisle Inlet</b><br>"
            f"Inlet Temp: {t_in[i]:.2f} °C<br>"
            f"ASHRAE Ceiling: {safe_max_c:.1f} °C<br>"
            f"Status: {'SLA VIOLATION' if t_in[i] > safe_max_c else 'NOMINAL'}"
            for i in range(num_racks)
        ],
        [
            f"<b>{rack_labels[i]} — Compute Load</b><br>"
            f"CPU Utilization: {cpu_util[i] * 100:.1f}%<br>"
            f"IT Power Draw: {rack_it_kw[i]:.2f} kW"
            for i in range(num_racks)
        ],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=rack_labels,
            y=["Hot Aisle Exhaust", "Cold Aisle Inlet (SLA)", "CPU Load & Power"],
            text=text_matrix,
            texttemplate="%{text}",
            textfont={"size": 12, "color": "white"},
            hovertext=hover_matrix,
            hoverinfo="text",
            colorscale=[
                [0.0, "#0f766e"],
                [0.45, "#0284c7"],
                [0.75, "#d97706"],
                [1.0, "#dc2626"],
            ],
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(
                title="Severity",
                tickvals=[0.1, 0.5, 0.9],
                ticktext=["Cool / Low", "Nominal", "Hot / Peak"],
            ),
        )
    )

    fig.update_layout(
        title="Server Hall 2D Thermal & Workload Matrix",
        height=300,
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis=dict(side="top"),
    )
    return fig


def build_rack_profile_bar(
    t_in: np.ndarray,
    t_out: np.ndarray,
    safe_max_c: float = 27.0,
) -> go.Figure:
    rack_labels = [f"R{i:02d}" for i in range(len(t_in))]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Cold-Aisle Inlet (°C)",
            x=rack_labels,
            y=t_in,
            marker_color=[
                "#ef4444" if temp > safe_max_c else "#0ea5e9" for temp in t_in
            ],
        )
    )
    fig.add_trace(
        go.Bar(
            name="Hot-Aisle Exhaust (°C)",
            x=rack_labels,
            y=t_out,
            marker_color="#f97316",
            opacity=0.85,
        )
    )
    fig.add_hline(
        y=safe_max_c,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text=f"ASHRAE Max Inlet ({safe_max_c:.1f}°C)",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Per-Rack Thermal Gradient & Edge Recirculation Profile",
        barmode="group",
        yaxis_title="Temperature (°C)",
        yaxis_range=[14, max(40, float(np.max(t_out)) + 3.0)],
        height=300,
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
