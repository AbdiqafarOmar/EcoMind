import io, math, base64, requests
import numpy as np, pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_lottie import st_lottie
from fpdf import FPDF

# ---------- CONFIG ----------
st.set_page_config(page_title="EcoMind — AI Carbon Footprint Explorer", page_icon="🌱", layout="wide")

# ---------- STYLE ----------
st.markdown("""
<style>
.main-title {
    font-size: 2.3rem; font-weight:700;
    background:linear-gradient(90deg,#00c6ff,#0072ff);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
div[data-testid="metric-container"]{
    background:#1e1e1e;padding:12px;border-radius:12px;
    box-shadow:0 0 10px rgba(0,0,0,0.3);
}
[data-testid="stSidebar"]{background:#111111;color:white;}
button{background:linear-gradient(90deg,#00c6ff,#0072ff)!important;
    color:white!important;border-radius:8px!important;}
</style>
""", unsafe_allow_html=True)

# ---------- UTILS ----------
def safe_float(x, default=np.nan):
    try:
        return float(x) if x not in (None, "") else default
    except Exception:
        return default

def estimate_flops(p_b, t_b, mult=6.0):
    if any(pd.isna(v) for v in [p_b, t_b]): 
        return np.nan
    return mult * (p_b * 1e9) * (t_b * 1e9)

def energy_kwh_from_flops(flops, flops_j, pue=1.2):
    if any(pd.isna(v) for v in [flops, flops_j, pue]) or flops_j <= 0:
        return np.nan
    j = flops / flops_j
    return (j / 3_600_000) * pue

def co2e_from_kwh(kwh, grid): 
    return np.nan if pd.isna(kwh) else kwh * grid

def format_si(x):
    if pd.isna(x): 
        return "—"
    for s, u in [(1e18,"E"),(1e15,"P"),(1e12,"T"),(1e9,"G"),(1e6,"M"),(1e3,"k")]:
        if abs(x) >= s: 
            return f"{x/s:.2f}{u}"
    return f"{x:.2f}"

# ---------- HEADER ----------
st.markdown("<h1 class='main-title'>🌱 EcoMind — AI Carbon Footprint Explorer</h1>", unsafe_allow_html=True)
anim = requests.get("https://assets9.lottiefiles.com/packages/lf20_ydo1amjm.json").json()
st_lottie(anim, height=200, key="earth")

st.write("Estimate and compare **training compute, energy use, and CO₂e** of AI models. "
         "Adjust assumptions for hardware efficiency, datacenter PUE, and regional grid intensity. "
         "Values are **estimates** for comparative insight.")

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("Assumptions")
    preset = st.selectbox(
        "Hardware preset (FLOPs/J):",
        [
            "Custom",
            "Modern GPU ~ 1.5e11",
            "Older GPU ~ 6.0e10",
            "TPU ~ 2.0e11"
        ],
        index=1
    )

    if "Modern" in preset:
        flops_j = 1.5e11
    elif "Older" in preset:
        flops_j = 6.0e10
    elif "TPU" in preset:
        flops_j = 2.0e11
    else:
        flops_j = safe_float(st.text_input("FLOPs per Joule (η)", value="1.50e11"))

    pue = safe_float(st.text_input("Datacenter PUE", value="1.20"))

    st.markdown("**Region (Grid Intensity)**")
    region_options = {
        "Global Avg (0.35)": 0.35,
        "US West (0.25)": 0.25,
        "Europe (0.20)": 0.20,
        "China (0.55)": 0.55,
        "India (0.70)": 0.70
    }
    region_choice = st.selectbox("Select Region", list(region_options.keys()))
    grid_intensity = region_options[region_choice]

    k_mult = safe_float(st.text_input("Training FLOPs Multiplier", value="6.0"))
    inference_mode = st.checkbox("Include Inference Energy Estimate", value=False)
    if inference_mode:
        hours = st.number_input("Daily Inference Hours", 0.0, 24.0, 4.0)
        days = st.number_input("Days Per Year", 0, 365, 200)
        inf_kwh = hours * days * 0.8
        st.caption(f"Estimated Inference Energy: ≈ {inf_kwh:.1f} kWh/year")
    else:
        inf_kwh = 0

# ---------- TABS ----------
tab1, tab2, tab3 = st.tabs(["Single Model", "Batch Compare (CSV)", "Assumptions & Notes"])

# ---------- SINGLE MODEL ----------
with tab1:
    st.subheader("Single Model Estimator")
    c1, c2, c3 = st.columns(3)
    model = c1.text_input("Model Name", "MyLLM-7B")
    p_b = c2.number_input("Parameters (B)", 0.0, 10000.0, 7.0)
    t_b = c3.number_input("Training Tokens (B)", 0.0, 10000.0, 1.0)

    if st.button("Estimate (Single)"):
        fl = estimate_flops(p_b, t_b, k_mult)
        kwh = energy_kwh_from_flops(fl, flops_j, pue) + inf_kwh
        co2 = co2e_from_kwh(kwh, grid_intensity)
        m1, m2, m3 = st.columns(3)
        m1.metric("Training FLOPs", format_si(fl) + " FLOPs")
        m2.metric("Energy (kWh)", f"{kwh:,.2f}")
        m3.metric("CO₂e (kg)", f"{co2:,.2f}")
        st.dataframe(pd.DataFrame([{
            "Model": model,
            "Params (B)": p_b,
            "Tokens (B)": t_b,
            "FLOPs (η)": flops_j,
            "PUE": pue,
            "Region kg/kWh": grid_intensity,
            "Inference kWh": inf_kwh,
            "Total kWh": kwh,
            "CO₂e (kg)": co2
        }]))

# ---------- BATCH COMPARE ----------
with tab2:
    st.subheader("Batch Compare (Upload CSV)")
    st.caption("CSV columns: model, params_billion, train_tokens_billion")
    up = st.file_uploader("Upload CSV", type=["csv"])
    demo = st.checkbox("Use example dataset", value=True)
    df = None

    if demo and not up:
        demo_csv = io.StringIO(
            "model,params_billion,train_tokens_billion\n"
            "Alpha-7B,7,1.2\n"
            "Beta-13B,13,2.0\n"
            "Gamma-70B,70,1.0\n"
            "Delta-34B,34,1.5\n"
        )
        df = pd.read_csv(demo_csv)
    elif up:
        df = pd.read_csv(up)

    if df is not None:
        df["FLOPs"] = df.apply(lambda r: estimate_flops(r["params_billion"], r["train_tokens_billion"], k_mult), axis=1)
        df["kWh"] = df["FLOPs"].apply(lambda x: energy_kwh_from_flops(x, flops_j, pue) + inf_kwh)
        df["CO2e (kg)"] = df["kWh"].apply(lambda x: co2e_from_kwh(x, grid_intensity))
        st.dataframe(df)

        # Plotly charts
        bar = px.bar(df, x="model", y="CO2e (kg)", color="CO2e (kg)",
                     color_continuous_scale="Blues", title="CO₂e by Model (Estimated)")
        bar.update_layout(title_x=0.5, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(bar, use_container_width=True)

        scatter = px.scatter(df, x="params_billion", y="CO2e (kg)",
                             size="train_tokens_billion", color="CO2e (kg)", hover_name="model",
                             color_continuous_scale="Viridis", title="Parameters vs CO₂e")
        scatter.update_layout(title_x=0.5, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(scatter, use_container_width=True)

        # CSV download
        csv = df.to_csv(index=False).encode()
        st.download_button("📥 Download Results CSV", csv, "EcoMind_results.csv", "text/csv")

        # PDF report
        if st.button("📄 Generate PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "EcoMind Summary Report", ln=1)
            pdf.set_font("Arial", "", 12)
            for _, r in df.iterrows():
                pdf.cell(0, 8, f"{r['model']}: {r['CO2e (kg)']:.2f} kg CO₂e", ln=1)
            b64 = base64.b64encode(pdf.output(dest="S").encode("latin1")).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="EcoMind_Report.pdf">Click to Download PDF Report</a>'
            st.markdown(href, unsafe_allow_html=True)

# ---------- NOTES ----------
with tab3:
    st.subheader("Assumptions & Methodology")
    st.markdown("""
- **Training FLOPs ≈ multiplier × parameters × training tokens**
- **Energy (kWh)** = FLOPs ÷ η × (PUE / 3.6×10⁶)
- **CO₂e (kg)** = kWh × grid intensity
- Adjustable inputs enable transparent scenario analysis.
- Next steps: region automation, inference profiling, embodied hardware carbon.
""")
