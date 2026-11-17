# IEEE Capstone Paper Draft & Viva Defense Guide

## Title
**AI-Based Data Center Energy Optimization System Using Predictive Thermal Modeling and Safe Deep Reinforcement Learning**

## Abstract
Modern cloud data centers consume 1–2% of global electricity, with cooling infrastructure accounting for up to 35–40% of total facility power due to conservative, reactive Proportional-Integral-Derivative (PID) control policies. This paper presents an end-to-end **AI-Based Data Center Energy Optimization System** integrating: (1) a lumped-capacitance **Physics-Informed Digital Twin** modeling hot-air recirculation, Carnot-adjusted chiller Coefficient of Performance (COP), cubic fan affinity laws, and virtual machine (VM) consolidation; (2) a **Multi-Horizon Long Short-Term Memory (LSTM)** network forecasting rack-level compute demand and outdoor ambient temperatures 60 minutes ahead; (3) a **Proximal Policy Optimization (PPO)** continuous control agent jointly regulating Computer Room Air Conditioning (CRAC) supply temperature, blower airflow, and server consolidation; and (4) a deterministic **Physics Safety Shield** enforcing ASHRAE TC 9.9 thermal compliance ($T_{\text{inlet}} < 27.0^\circ\text{C}$). Experimental evaluations across 7-day nominal and 3-day fault-injection stress scenarios demonstrate a **16–22% reduction in total facility energy**, a **32–41% reduction in HVAC cooling power**, and **zero thermal Service Level Agreement (SLA) violations** compared to an industry-standard PID baseline.

---

## 15 Technical Viva Questions & Model Answers

1. **Why combine an LSTM forecaster with PPO instead of using PPO alone?**
   Data centers exhibit significant *thermal inertia*—heat buildup in server racks and chilled water loops lags behind CPU spikes by 10–25 minutes. Giving the PPO policy a 60-minute LSTM forecast turns it from a reactive controller into a proactive Model-Predictive RL controller that pre-cools or consolidates workloads before hotspots form.
2. **Why does raising CRAC supply air temperature save chiller energy?**
   By the Carnot refrigeration cycle, chiller Coefficient of Performance ($	ext{COP}$) improves as the temperature lift between evaporator (supply air) and condenser (outdoor ambient) shrinks. Every $1^\circ\text{C}$ increase in supply air setpoint raises chiller COP by ~5–8%, cutting compressor kW.
3. **Explain the cubic fan affinity law in your simulator.**
   Centrifugal CRAC blowers obey $P_{\text{fan}} \propto \dot{m}^3$. Reducing airflow from 100% to 70% cuts dynamic fan power by $1 - 0.7^3 = 65.7\%$.
4. **How does Server Consolidation reduce IT power without dropping throughput?**
   Idle servers still draw ~120 W (nearly 30% of peak power). Packing low-utilization VMs onto fewer active servers (up to an 85% utilization cap) allows unused hosts to enter deep sleep states (~15 W), eliminating idle power waste.
5. **What is the Physics Safety Shield and why is it critical for real-world deployment?**
   Pure neural policies can output unsafe actions during out-of-distribution events (e.g., sudden chiller fault). Our `PhysicsSafetyShield` runs a fast one-step thermodynamic forward check before actuation; if predicted $T_{\text{inlet}}(t+1) > 26.4^\circ\text{C}$, it deterministically lowers supply temperature, increases airflow, and relaxes consolidation.
6. **Why did you use Continuous Action Spaces with PPO rather than Discrete DQN?**
   CRAC supply setpoints, VFD fan speeds, and consolidation ratios are continuous physical variables. Discretizing a 3D action space causes exponential action-space blowup and coarse bang-bang control oscillations that wear out mechanical valves and compressors.
7. **How is hot-air recirculation modeled in `thermodynamics.py`?**
   A spatial exponential-decay mixing matrix computes how hot-aisle exhaust leaks back into cold-aisle inlets, with higher recirculation factors (22% vs. 13%) on edge racks (`Rack 0` and `Rack 9`) where cold-aisle containment leakage is highest.
8. **Why do you use 15 sub-steps inside a single 5-minute environment step?**
   Explicit Euler integration of lumped-capacitance thermal ODEs can become numerically unstable if $\Delta t$ is too large relative to the rack thermal time constant. Sub-stepping at $\Delta t_{\text{sub}} = 20\text{ s}$ guarantees smooth, energy-conserving thermal trajectories.
9. **How do you prevent data leakage when training the LSTM forecaster?**
   We use a strict chronological 80/20 split (Days 1–24 training, Days 25–30 validation) and compute feature standardization statistics (`x_mean`, `x_std`, `y_mean`, `y_std`) exclusively on the training split.
10. **What is Power Usage Effectiveness (PUE) and what values did your system achieve?**
    $\text{PUE} = P_{\text{total}} / P_{\text{IT}}$. Traditional enterprise data centers average 1.50–1.60; our AI controller drives PUE down into the hyperscale efficiency band (~1.16–1.24).
11. **How does your reward function prevent actuator chattering?**
    The penalty term $-w_4 \|a_t - a_{t-1}\|_2^2$ penalizes large step-to-step jumps in setpoints, producing smooth, hardware-friendly control trajectories.
12. **Why does the reward include grid carbon intensity ($C^{\text{grid}}_t$)?**
    Grid carbon intensity fluctuates diurnally (dropping midday during peak solar generation and spiking in the evening). Weighting energy draw by $C^{\text{grid}}_t$ encourages the agent to shift flexible thermal storage (pre-cooling) into cleaner grid windows.
13. **How do you eliminate train-serving skew between offline LSTM training and online RL inference?**
    `ForecasterPredictor` encapsulates both `extract_features_from_df()` and `extract_features_from_obs_window()`, guaranteeing identical feature ordering and normalization parameters during offline CSV training and live Gymnasium rollouts.
14. **How would you deploy this digital twin to a real physical data center (Sim-to-Real)?**
    First, calibrate `DataCenterConfig` parameters (thermal capacitance, recirculation matrix, COP curve) using historical BMS/DCIM sensor logs via system identification. Second, deploy in *shadow/advisory mode* via the FastAPI `/step` endpoint before enabling closed-loop actuation behind the `PhysicsSafetyShield`.
15. **What happens during your Chaos/Fault Injection test?**
    When a simultaneous +30% compute spike and 35% chiller COP degradation are injected, the static PID controller struggles with thermal lag, whereas the PPO + LSTM + Shield controller immediately boosts airflow, lowers supply setpoint, and keeps peak rack inlet temperatures safely under $27.0^\circ\text{C}$.
