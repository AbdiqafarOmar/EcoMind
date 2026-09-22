import pandas as pd
import pytest

from ecomind_engine import JOULES_PER_KWH, SCENARIO_PRESETS, Scenario, compare_scenarios, estimate_batch, estimate_training


def test_training_flops_uses_scaling_law():
    result = estimate_training("test", 7, 1000, SCENARIO_PRESETS["Baseline"])
    assert result.training_flops == pytest.approx(6 * 7e9 * 1000e9)


def test_energy_conversion_and_pue():
    scenario = Scenario("unit", 1, 1, 2, 0.5)
    result = estimate_training("unit", 1, 1, scenario)
    assert result.it_energy_kwh == pytest.approx(1e18 / JOULES_PER_KWH)
    assert result.facility_energy_kwh == pytest.approx(result.it_energy_kwh * 2)
    assert result.operational_co2e_kg == pytest.approx(result.facility_energy_kwh * 0.5)


@pytest.mark.parametrize("params,tokens", [(0, 1), (-1, 1), (1, 0), (1, -1)])
def test_workload_inputs_must_be_positive(params, tokens):
    with pytest.raises(ValueError):
        estimate_training("bad", params, tokens, SCENARIO_PRESETS["Baseline"])


@pytest.mark.parametrize("field", ["compute_multiplier", "efficiency_flops_per_joule", "pue", "grid_kg_co2e_per_kwh"])
def test_scenario_inputs_must_be_positive(field):
    values = dict(name="bad", compute_multiplier=6, efficiency_flops_per_joule=1e11, pue=1.2, grid_kg_co2e_per_kwh=0.35)
    values[field] = 0
    with pytest.raises(ValueError):
        Scenario(**values).validate()


def test_batch_requires_schema():
    with pytest.raises(ValueError, match="Missing required columns"):
        estimate_batch(pd.DataFrame({"model": ["x"]}), SCENARIO_PRESETS["Baseline"])


def test_batch_rejects_non_numeric_values():
    frame = pd.DataFrame({"model": ["x"], "params_billion": ["seven"], "train_tokens_billion": [1000]})
    with pytest.raises(ValueError, match="invalid values"):
        estimate_batch(frame, SCENARIO_PRESETS["Baseline"])


def test_batch_returns_auditable_columns():
    frame = pd.DataFrame({"model": ["x"], "params_billion": [7], "train_tokens_billion": [1000]})
    result = estimate_batch(frame, SCENARIO_PRESETS["Baseline"])
    assert len(result) == 1
    assert {"training_flops", "it_energy_kwh", "facility_energy_kwh", "operational_co2e_kg"} <= set(result)


def test_scenario_comparison_orders_estimates():
    result = compare_scenarios("x", 7, 1000)
    values = result.set_index("scenario").operational_co2e_kg
    assert values["Efficient"] < values["Baseline"] < values["Conservative"]
