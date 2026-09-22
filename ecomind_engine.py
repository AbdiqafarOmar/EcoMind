"""Transparent estimation engine for AI compute, energy, and operational carbon."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd

JOULES_PER_KWH = 3_600_000
REQUIRED_COLUMNS = ("model", "params_billion", "train_tokens_billion")


@dataclass(frozen=True)
class Scenario:
    name: str
    compute_multiplier: float = 6.0
    efficiency_flops_per_joule: float = 1.5e11
    pue: float = 1.20
    grid_kg_co2e_per_kwh: float = 0.35

    def validate(self) -> None:
        values = asdict(self)
        for key in ("compute_multiplier", "efficiency_flops_per_joule", "pue", "grid_kg_co2e_per_kwh"):
            if values[key] <= 0:
                raise ValueError(f"{key} must be greater than zero")


@dataclass(frozen=True)
class Estimate:
    model: str
    params_billion: float
    train_tokens_billion: float
    training_flops: float
    it_energy_kwh: float
    facility_energy_kwh: float
    operational_co2e_kg: float
    scenario: str

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


SCENARIO_PRESETS = {
    "Efficient": Scenario("Efficient", 6.0, 2.0e11, 1.10, 0.20),
    "Baseline": Scenario("Baseline", 6.0, 1.5e11, 1.20, 0.35),
    "Conservative": Scenario("Conservative", 6.0, 6.0e10, 1.40, 0.55),
}


def estimate_training(model: str, params_billion: float, tokens_billion: float,
                      scenario: Scenario) -> Estimate:
    scenario.validate()
    if params_billion <= 0 or tokens_billion <= 0:
        raise ValueError("Parameters and training tokens must be greater than zero")
    flops = scenario.compute_multiplier * params_billion * 1e9 * tokens_billion * 1e9
    it_energy_kwh = flops / scenario.efficiency_flops_per_joule / JOULES_PER_KWH
    facility_energy_kwh = it_energy_kwh * scenario.pue
    co2e_kg = facility_energy_kwh * scenario.grid_kg_co2e_per_kwh
    return Estimate(model.strip() or "Unnamed model", float(params_billion), float(tokens_billion),
                    flops, it_energy_kwh, facility_energy_kwh, co2e_kg, scenario.name)


def estimate_batch(frame: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("The uploaded CSV is empty")
    if len(frame) > 500:
        raise ValueError("Batch uploads are limited to 500 rows")
    clean = frame.loc[:, REQUIRED_COLUMNS].copy()
    for column in ("params_billion", "train_tokens_billion"):
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
        if clean[column].isna().any():
            rows = (clean.index[clean[column].isna()] + 2).tolist()
            raise ValueError(f"{column} contains invalid values on CSV rows {rows}")
        if (clean[column] <= 0).any():
            rows = (clean.index[clean[column] <= 0] + 2).tolist()
            raise ValueError(f"{column} must be positive on CSV rows {rows}")
    clean["model"] = clean["model"].fillna("").astype(str).str.strip()
    clean.loc[clean["model"] == "", "model"] = "Unnamed model"
    estimates = [estimate_training(row.model, row.params_billion, row.train_tokens_billion, scenario)
                 for row in clean.itertuples(index=False)]
    return pd.DataFrame(estimate.to_dict() for estimate in estimates)


def compare_scenarios(model: str, params_billion: float, tokens_billion: float,
                      scenarios: Iterable[Scenario] | None = None) -> pd.DataFrame:
    selected = list(scenarios or SCENARIO_PRESETS.values())
    return pd.DataFrame(estimate_training(model, params_billion, tokens_billion, scenario).to_dict()
                        for scenario in selected)


def format_scientific(value: float) -> str:
    return f"{value:.3e}"
