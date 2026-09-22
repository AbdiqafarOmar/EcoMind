# EcoMind

EcoMind is a transparent scenario-analysis application for estimating AI training compute, IT energy, facility energy, and operational carbon emissions. It is designed to make every assumption visible, testable, and exportable.

> EcoMind produces decision-support estimates. It does not claim to measure actual emissions or provide a complete lifecycle assessment without workload and infrastructure telemetry.

## What it does

- Estimates training FLOPs from model parameters, training tokens, and a configurable compute multiplier.
- Converts compute into IT electricity using an explicit effective FLOPs-per-joule assumption.
- Applies datacenter PUE and grid-carbon intensity to estimate facility energy and operational CO2e.
- Compares efficient, baseline, and conservative scenarios to expose assumption sensitivity.
- Validates CSV uploads and analyzes up to 500 model configurations at once.
- Exports reproducible CSV, JSON, and PDF evidence artifacts.
- Separates the calculation engine, reporting layer, and Streamlit interface.
- Runs 14 automated tests in GitHub Actions.

## Calculation model

```text
training FLOPs = multiplier x parameters x training tokens
IT energy (kWh) = FLOPs / (effective FLOPs/J x 3,600,000)
facility energy = IT energy x PUE
operational CO2e = facility energy x grid intensity
```

The included infrastructure presets are illustrative scenarios, not claims about specific hardware or regions. Users can replace every value with documented assumptions.

### Boundary

Included:

- Estimated training compute
- IT electricity
- Datacenter overhead through PUE
- Location-based operational electricity emissions

Excluded:

- Embodied hardware emissions
- Networking and storage
- Experimentation and failed training runs
- Datacenter construction
- Inference
- Supply-chain impacts

## Run locally

```bash
git clone https://github.com/AbdiqafarOmar/EcoMind.git
cd EcoMind
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Test

```bash
pytest -q
```

## CSV schema

```csv
model,params_billion,train_tokens_billion
Alpha-7B,7,1000
Beta-13B,13,1500
```

Rows must contain positive numeric values. Uploads are capped at 500 rows so failures remain explainable in the interface.

## Architecture

```text
streamlit_app.py      interface, charts, scenario controls
ecomind_engine.py     validated calculation and batch-analysis engine
reporting.py          CSV/JSON/PDF evidence exports
sample_models.csv     deterministic demonstration data
tests/                automated engine tests
```

## Responsible use

EcoMind is most useful for comparative analysis, sensitivity testing, and communicating how infrastructure assumptions influence an estimate. Do not present results as audited carbon measurements unless the inputs are backed by measured workload, hardware, datacenter, and electricity data.
