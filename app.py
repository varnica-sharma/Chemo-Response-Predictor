import streamlit as st
import torch
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from model import IC50Net

# Load assets
torch.serialization.add_safe_globals({"IC50Net": IC50Net})
model = torch.load("models/ic50_model.pt", weights_only=False)
model.eval()

scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/onehot_encoder.pkl")
label_enc_drug = joblib.load("models/drug_encoder.pkl")
label_enc_cell = joblib.load("models/cell_encoder.pkl")

# Setup page
st.set_page_config(
    page_title="Personalized Chemotherapy Response Predictor",
    page_icon="💊",
    layout="wide"
)

# Title
st.title("💊 Personalized Chemotherapy Response Prediction using Genomic & Drug Metadata (GDSC)")

# Sidebar inputs
st.sidebar.header("🧬 Input Patient & Drug Info")

# Drug and cell line
drug_name = st.sidebar.selectbox("Drug Name", label_enc_drug.classes_)
cell_line = st.sidebar.selectbox("Cell Line", label_enc_cell.classes_)
tissue = st.sidebar.selectbox("Tissue Type", ['breast', 'lung', 'skin', 'bone', 'blood'])
tcga = st.sidebar.selectbox("TCGA Classification", ['brca', 'luad', 'skcm', 'lusc', 'gbm'])

# Predict button
if st.sidebar.button("🔮 Predict IC50"):

    # Encode inputs
    drug_enc = label_enc_drug.transform([drug_name])[0]
    cell_enc = label_enc_cell.transform([cell_line])[0]

    # Dummy values for hidden features
    z_score = 0.0
    max_conc = 10.0
    z_scaled, max_scaled = scaler.transform([[z_score, max_conc]])[0]

    # One-hot encode metadata
    meta_df = pd.DataFrame([[tissue.lower(), tcga.lower()]], columns=["Tissue", "TCGA_Classification"])
    meta_onehot = encoder.transform(meta_df)

    # Create input tensor
    x = np.hstack([[cell_enc, drug_enc, z_scaled, max_scaled], meta_onehot[0]]).astype(np.float32)
    x_tensor = torch.tensor([x], dtype=torch.float32)

    # Predict IC50
    with torch.no_grad():
        pred = model(x_tensor).item()

    # Display Result (Stylish)
    st.markdown("### 🎯 Predicted Drug Sensitivity")
    st.markdown(f"""
    <div style='
        background-color: #174D30;
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 1.8rem;
    '>
    💡 Predicted IC50: <strong>{round(pred, 4)}</strong>
    </div>
    """, unsafe_allow_html=True)

    # Interpretation Section
    st.markdown("### 🧠 Interpretation of Prediction")
    if pred <= 4:
        interpretation = "high sensitivity"
        note = "This means the drug is likely to be effective at lower concentrations."
    elif 4 < pred <= 6:
        interpretation = "moderate sensitivity"
        note = "Effectiveness may vary; dosage adjustment might be needed."
    else:
        interpretation = "low sensitivity"
        note = "The drug might not be effective at standard dosages."

    st.markdown(f"""
    <div style='background-color: #174D30; padding: 1rem; border-radius: 10px; color: white;'>
    <b>🔷 The predicted IC50 indicates <u>{interpretation}</u> to the selected drug.</b><br>
    {note}
    </div>
    """, unsafe_allow_html=True)

    # Plotly-based Interpretation Chart
    x_vals = [0, 4, 6, 10]
    colors = ["#b6e3a8", "#fffac8", "#ffb3b3"]
    labels = ["Sensitive", "Moderate", "Resistant"]

    fig = go.Figure()
    for i in range(3):
        fig.add_shape(
            type="rect",
            x0=x_vals[i], x1=x_vals[i + 1], y0=0, y1=1,
            fillcolor=colors[i],
            line=dict(width=0),
            layer='below'
        )
        fig.add_annotation(
            x=(x_vals[i] + x_vals[i + 1]) / 2,
            y=0.5,
            text=labels[i],
            showarrow=False,
            font=dict(size=14)
        )

    fig.add_vline(
        x=pred,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Predicted = {round(pred, 2)}",
        annotation_position="top"
    )

    fig.update_yaxes(visible=False)
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=20, b=20),
        title="IC50 Sensitivity Interpretation",
        plot_bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("🔬 Powered by PyTorch • Streamlit • GDSC")
