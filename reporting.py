"""Export helpers for reproducible EcoMind audit artifacts."""

import json
import pandas as pd
from fpdf import FPDF


def dataframe_to_json(frame: pd.DataFrame) -> bytes:
    return json.dumps(frame.to_dict(orient="records"), indent=2).encode("utf-8")


def dataframe_to_pdf(frame: pd.DataFrame, title: str = "EcoMind Estimate Report") -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, "Scenario estimates only. Results depend on user-supplied compute, hardware, datacenter, and electricity-grid assumptions.")
    pdf.ln(2)
    for record in frame.to_dict(orient="records"):
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, str(record.get("model", "Model")), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        lines = (
            f"Scenario: {record.get('scenario', '')}",
            f"Training FLOPs: {record.get('training_flops', 0):.3e}",
            f"Facility energy: {record.get('facility_energy_kwh', 0):,.3f} kWh",
            f"Operational CO2e: {record.get('operational_co2e_kg', 0):,.3f} kg",
        )
        for line in lines:
            pdf.cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
    output = pdf.output()
    return bytes(output) if not isinstance(output, bytes) else output
