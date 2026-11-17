# System Architecture & Mathematical Formulation

## 1. End-to-End Architecture

```mermaid
flowchart TB
    subgraph L1["1. Digital Twin & Physical Plant Layer"]
        IT["Multi-Rack Server Hall (N=10 Racks, 200 Servers)"]
        CRAC["CRAC & Chiller Plant (Supply Air, Airflow, Carnot COP)"]
        ENV["External Environment & Grid (Ambient Weather, gCO2/kWh)"]
        ENV --> CRAC
        IT <-->|"Convective Heat Exchange & Recirculation"| CRAC
    end

    subgraph L2["2. Telemetry & Predictive Forecasting Layer"]
        SENS["Telemetry Collector (5-Min Sampling)"]
        HIST["Rolling State Buffer (24 Steps = 2 Hours)"]
        LSTM["Multi-Horizon LSTM Forecaster (+15 to +60 Min Lookahead)"]
        IT & CRAC & ENV --> SENS --> HIST --> LSTM
    end

    subgraph L3["3. AI Optimization & Safety Guardrail Layer"]
        STATE["Composite State Vector s_t"]
        RL["DRL Policy Agent (PPO Continuous Control)"]
        SHIELD["Physics Safety Shield (ASHRAE < 27.0 °C Hard Filter)"]
        SENS & LSTM --> STATE --> RL --> SHIELD
        SHIELD -->|"CRAC Setpoints & VM Consolidation"| IT & CRAC
    end

    subgraph L4["4. Production Serving & Command Center"]
        API["FastAPI REST Microservice (/step, /metrics, /inject-fault)"]
        UI["Multi-Page Streamlit Command Center Dashboard"]
        SHIELD & SENS --> API --> UI
    end
```

## 2. Markov Decision Process (MDP) Equations

- **Power Usage Effectiveness:** $\text{PUE}_t = \frac{P^{\text{IT}}_t + P^{\text{cool}}_t}{P^{\text{IT}}_t}$
- **Fan Affinity Law:** $P_{\text{fan}}(\dot{m}_t) = P_{\text{fan,max}} \cdot \dot{m}_t^3 + P_{\text{pump,base}}$
- **Multi-Objective Reward:**
  $$R_t = -\left( w_1 \frac{P^{\text{total}}_t}{P^{\text{max}}} + w_2 \frac{P^{\text{total}}_t \cdot C^{\text{grid}}_t}{E^{\text{carbon}}_{\text{norm}}} + w_3 \sum_{i=1}^N \left[\max(0, T^{\text{in}}_{i,t+1} - 27.0)^2 + 50\,\mathbb{I}(T^{\text{in}}_{i,t+1} \ge 30.0)\right] + w_4 \|a_t - a_{t-1}\|_2^2 \right)$$
