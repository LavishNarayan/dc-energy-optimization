import numpy as np
from src.simulator.thermodynamics import DataCenterConfig, DataCenterPhysics
from src.simulator.workload_gen import TelemetryProfileGenerator


def test_fan_affinity_cubic_law():
    cfg = DataCenterConfig(num_racks=4)
    plant = DataCenterPhysics(cfg)
    plant.reset(t_supply_init=18.0)

    util = np.full(4, 0.5)
    m_low = plant.step(util, t_ambient_c=25.0, t_supply_c=18.0, airflow_frac=0.5, consolidation_ratio=0.0)
    plant.reset(t_supply_init=18.0)
    m_high = plant.step(util, t_ambient_c=25.0, t_supply_c=18.0, airflow_frac=1.0, consolidation_ratio=0.0)

    expected_low = cfg.p_fan_max_kw * (0.5 ** 3) + cfg.p_pump_base_kw
    expected_high = cfg.p_fan_max_kw * (1.0 ** 3) + cfg.p_pump_base_kw
    assert np.isclose(m_low["fan_power_kw"], expected_low)
    assert np.isclose(m_high["fan_power_kw"], expected_high)


def test_server_consolidation_reduces_idle_draw():
    cfg = DataCenterConfig(num_racks=4, servers_per_rack=20)
    plant = DataCenterPhysics(cfg)
    util = np.full(4, 0.25)

    p_unconsolidated, active_0 = plant.compute_it_power(util, consolidation_ratio=0.0)
    p_consolidated, active_1 = plant.compute_it_power(util, consolidation_ratio=1.0)

    assert active_1 < active_0
    assert p_consolidated.sum() < p_unconsolidated.sum()


def test_workload_generator_shapes():
    gen = TelemetryProfileGenerator(num_racks=10, dt_minutes=5, seed=7)
    data = gen.generate(num_steps=288)
    assert data["cpu_util"].shape == (288, 10)
    assert data["t_ambient"].shape == (288,)
    assert data["carbon_intensity"].shape == (288,)
    assert np.all((data["cpu_util"] >= 0.0) & (data["cpu_util"] <= 1.0))
