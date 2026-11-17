from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_pue_gauge(current_pue: float, baseline_pue: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=current_pue,
            number={"valueformat": ".3f", "font": {"size": 36}},
            delta={
                "reference": baseline_pue,
                "increasing": {"color": "#ef4444"},
                "decreasing": {"color": "#10b981"},
                "valueformat": ".3f",
            },
            title={"text": "Live AI PUE (vs. PID Baseline)", "font": {"size": 15}},
            gauge={
                "axis": {"range": [1.0, 1.85], "tickwidth": 1},
                "bar": {"color": "#0284c7"},
                "steps": [
                    {"range": [1.0, 1.25], "color": "rgba(16, 185, 129, 0.25)"},
                    {"range": [1.25, 1.45], "color": "rgba(245, 158, 11, 0.22)"},
                    {"range": [1.45, 1.85], "color": "rgba(239, 68, 68, 0.22)"},
                ],
                "threshold": {
                    "line": {"color": "#ef4444", "width": 3},
                    "thickness": 0.75,
                    "value": baseline_pue,
                },
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=45, b=10))
    return fig


def build_power_split_chart(history_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=history_df["sim_time"],
            y=history_df["it_power_kw"],
            name="IT Server Load (kW)",
            stackgroup="power",
            mode="lines",
            line=dict(width=0.5, color="#3b82f6"),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=history_df["sim_time"],
            y=history_df["chiller_power_kw"],
            name="Chiller Plant (kW)",
            stackgroup="power",
            mode="lines",
            line=dict(width=0.5, color="#06b6d4"),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=history_df["sim_time"],
            y=history_df["fan_power_kw"],
            name="CRAC Fans & Pumps (kW)",
            stackgroup="power",
            mode="lines",
            line=dict(width=0.5, color="#8b5cf6"),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=history_df["sim_time"],
            y=history_df["pid_total_power_kw"],
            name="PID Total Power (kW)",
            mode="lines",
            line=dict(color="#94a3b8", width=2, dash="dot"),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=history_df["sim_time"],
            y=history_df["max_inlet_temp_c"],
            name="Peak Inlet Temp (°C)",
            mode="lines",
            line=dict(color="#f97316", width=2.5),
        ),
        secondary_y=True,
    )

    fig.add_hline(
        y=27.0,
        line_dash="dash",
        line_color="#dc2626",
        secondary_y=True,
        annotation_text="27°C SLA Limit",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title="Real-Time Power Breakdown (AI vs. PID) & Peak Cold-Aisle Temperature",
        height=360,
        margin=dict(l=10, r=10, t=45, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Power (kW)", secondary_y=False)
    fig.update_yaxes(title_text="Inlet Temp (°C)", range=[15, 33], secondary_y=True)
    return fig


def build_actuator_telemetry_chart(history_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=history_df["sim_time"],
            y=history_df["t_supply_c"],
            name="CRAC Supply Setpoint (°C)",
            mode="lines",
            line=dict(color="#0ea5e9", width=2.2),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=history_df["sim_time"],
            y=history_df["airflow_frac"] * 100.0,
            name="CRAC Airflow (%)",
            mode="lines",
            line=dict(color="#10b981", width=2),
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=history_df["sim_time"],
            y=history_df["consolidation_ratio"] * 100.0,
            name="Server Consolidation (%)",
            mode="lines",
            line=dict(color="#a855f7", width=2, dash="dash"),
        ),
        secondary_y=True,
    )

    overrides = history_df[history_df["shield_override"] == 1]
    if not overrides.empty:
        fig.add_trace(
            go.Scatter(
                x=overrides["sim_time"],
                y=overrides["t_supply_c"],
                name="Safety Shield Override",
                mode="markers",
                marker=dict(color="#ef4444", size=9, symbol="diamond"),
            ),
            secondary_y=False,
        )

    fig.update_layout(
        title="AI Actuator Modulation & Safety Shield Interventions",
        height=320,
        margin=dict(l=10, r=10, t=45, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Supply Temp (°C)", range=[15, 25], secondary_y=False)
    fig.update_yaxes(title_text="Actuator %", range=[0, 105], secondary_y=True)
    return fig


def build_benchmark_comparison_chart(summary_df: pd.DataFrame, scenario: str) -> go.Figure:
    sub = summary_df[summary_df["Scenario"] == scenario]
    controllers = sub["Controller"].tolist()

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Energy Consumption (kWh)", "Carbon Emissions (kg CO2)"),
    )

    fig.add_trace(
        go.Bar(
            name="Total Energy (kWh)",
            x=controllers,
            y=sub["Total Energy (kWh)"],
            marker_color="#3b82f6",
            text=sub["Total Energy (kWh)"],
            textposition="auto",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            name="Cooling Energy (kWh)",
            x=controllers,
            y=sub["Cooling Energy (kWh)"],
            marker_color="#06b6d4",
            text=sub["Cooling Energy (kWh)"],
            textposition="auto",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            name="CO2 Emissions (kg)",
            x=controllers,
            y=sub["CO2 Emissions (kg)"],
            marker_color="#10b981",
            text=sub["CO2 Emissions (kg)"],
            textposition="auto",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=f"Benchmark Comparison — {scenario}",
        barmode="group",
        height=370,
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
    )
    return fig
