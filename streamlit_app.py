"""Streamlit interface for the EcoMind AI carbon scenario explorer."""

import pandas as pd
import plotly.express as px
import streamlit as st

from ecomind_engine import SCENARIO_PRESETS, Scenario, compare_scenarios, estimate_batch, estimate_training, format_scientific
from reporting import dataframe_to_json, dataframe_to_pdf

st.set_page_config(page_title="EcoMind | AI Carbon Scenario Explorer", page_icon="🌱", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1240px;}
.hero {padding: 1.4rem 1.6rem; border: 1px solid #d9e5dd; border-radius: 18px; background: linear-gradient(135deg, #f2fbf5 0%, #eef7ff 100%);}
.hero h1 {margin: 0; color: #123b2a; font-size: 2.25rem;}
.hero p {margin: .5rem 0 0; color: #35594a; font-size: 1.05rem;}
div[data-testid="stMetric"] {border: 1px solid #dfe7e2; padding: .75rem; border-radius: 12px; background: #ffffff;}
div[data-testid="stMetric"] [data-testid="stMetricLabel"],
div[data-testid="stMetric"] [data-testid="stMetricValue"] {color: #123b2a;}
.disclosure {font-size: .88rem; color: #53645c;}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="hero"><h1>EcoMind</h1><p>Transparent scenario analysis for AI training compute, facility energy, and operational carbon.</p></div>
""", unsafe_allow_html=True)
st.caption("Decision-support estimates, not measured emissions or a lifecycle assessment.")

with st.sidebar:
    st.header("Scenario controls")
    preset_name = st.selectbox("Infrastructure scenario", list(SCENARIO_PRESETS), index=1)
    preset = SCENARIO_PRESETS[preset_name]
    with st.expander("Edit assumptions", expanded=True):
        multiplier = st.number_input("Training compute multiplier", min_value=0.1, value=preset.compute_multiplier, step=0.5)
        efficiency = st.number_input("Effective compute efficiency (FLOPs/J)", min_value=1.0e6, value=preset.efficiency_flops_per_joule, format="%.2e")
        pue = st.number_input("Datacenter PUE", min_value=1.0, value=preset.pue, step=0.05)
        grid = st.number_input("Grid intensity (kg CO2e/kWh)", min_value=0.001, value=preset.grid_kg_co2e_per_kwh, step=0.01, format="%.3f")
    scenario = Scenario(preset_name, multiplier, efficiency, pue, grid)
    st.markdown('<p class="disclosure">All presets are illustrative. Replace them with documented workload and infrastructure values for formal analysis.</p>', unsafe_allow_html=True)

estimate_tab, batch_tab, sensitivity_tab, methodology_tab = st.tabs(["Model estimate", "Batch comparison", "Sensitivity", "Methodology"])

with estimate_tab:
    st.subheader("Estimate one training run")
    first, second, third = st.columns([1.5, 1, 1])
    model = first.text_input("Model label", "Example-7B")
    params = second.number_input("Parameters (billions)", min_value=0.01, value=7.0)
    tokens = third.number_input("Training tokens (billions)", min_value=0.01, value=1000.0)
    result = estimate_training(model, params, tokens, scenario)
    metrics = st.columns(4)
    metrics[0].metric("Training compute", format_scientific(result.training_flops) + " FLOPs")
    metrics[1].metric("IT energy", f"{result.it_energy_kwh:,.1f} kWh")
    metrics[2].metric("Facility energy", f"{result.facility_energy_kwh:,.1f} kWh")
    metrics[3].metric("Operational CO2e", f"{result.operational_co2e_kg:,.1f} kg")
    st.dataframe(pd.DataFrame([result.to_dict()]), use_container_width=True, hide_index=True)

with batch_tab:
    st.subheader("Compare a model portfolio")
    uploaded = st.file_uploader("Upload a CSV with model, params_billion, and train_tokens_billion", type="csv")
    source = uploaded if uploaded else "sample_models.csv"
    try:
        batch = estimate_batch(pd.read_csv(source), scenario)
    except (ValueError, pd.errors.ParserError) as exc:
        st.error(str(exc))
    else:
        st.dataframe(batch, use_container_width=True, hide_index=True)
        chart = px.bar(batch, x="model", y="operational_co2e_kg", color="facility_energy_kwh",
                       labels={"operational_co2e_kg": "Operational CO2e (kg)", "model": "Model"},
                       title="Estimated operational carbon by model", color_continuous_scale="Tealgrn")
        st.plotly_chart(chart, use_container_width=True)
        exports = st.columns(3)
        exports[0].download_button("Download CSV", batch.to_csv(index=False).encode(), "ecomind_estimates.csv", "text/csv")
        exports[1].download_button("Download JSON", dataframe_to_json(batch), "ecomind_estimates.json", "application/json")
        exports[2].download_button("Download PDF", dataframe_to_pdf(batch), "ecomind_report.pdf", "application/pdf")

with sensitivity_tab:
    st.subheader("Bound uncertainty with scenario comparison")
    sensitivity = compare_scenarios(model, params, tokens)
    st.dataframe(sensitivity, use_container_width=True, hide_index=True)
    sensitivity_chart = px.bar(sensitivity, x="scenario", y="operational_co2e_kg", color="scenario",
                               labels={"operational_co2e_kg": "Operational CO2e (kg)", "scenario": "Scenario"},
                               title="How infrastructure assumptions change the estimate")
    st.plotly_chart(sensitivity_chart, use_container_width=True)
    spread = sensitivity.operational_co2e_kg.max() / sensitivity.operational_co2e_kg.min()
    st.info(f"Across the included scenarios, the highest estimate is {spread:.1f}x the lowest estimate.")

with methodology_tab:
    st.subheader("Calculation boundary and equations")
    st.latex(r"\mathrm{Training\ FLOPs} = m \times N_{parameters} \times N_{tokens}")
    st.latex(r"\mathrm{IT\ energy\ (kWh)} = \frac{FLOPs}{FLOPs/J \times 3.6\times10^6}")
    st.latex(r"\mathrm{Facility\ energy} = \mathrm{IT\ energy} \times PUE")
    st.latex(r"\mathrm{Operational\ CO_2e} = \mathrm{Facility\ energy} \times \mathrm{grid\ intensity}")
    st.markdown("""
**Included:** estimated training compute, IT electricity, datacenter overhead through PUE, and location-based operational electricity emissions.

**Excluded:** embodied hardware emissions, networking, storage, cooling outside the PUE assumption, model experimentation, data-center construction, inference, and supply-chain impacts.

EcoMind exposes every assumption so results can be reproduced and challenged. It does not claim to measure a model's actual footprint without workload and infrastructure telemetry.
""")
