import streamlit as st
import torch
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
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

    # Interpretation of IC50
    st.markdown("### 🧠 Interpretation of Prediction")

    if pred < 4:
        st.success("🔹 The predicted IC50 indicates **high sensitivity** to the selected drug. This means the drug is likely to be effective at lower concentrations.")
    elif pred < 6:
        st.warning("🟡 The predicted IC50 indicates **moderate sensitivity**. The drug may require higher doses to be effective.")
    else:
        st.error("🔺 The predicted IC50 indicates **resistance**. The cancer cells are likely to be less responsive to this drug.")

    # Visual gauge bar
    fig, ax = plt.subplots(figsize=(6, 1.2))
    ax.axvspan(0, 4, color='green', alpha=0.4, label='Sensitive')
    ax.axvspan(4, 6, color='yellow', alpha=0.4, label='Moderate')
    ax.axvspan(6, 10, color='red', alpha=0.4, label='Resistant')
    ax.axvline(pred, color='black', linestyle='--', linewidth=2, label=f'Predicted = {round(pred, 2)}')

    ax.set_xlim(0, 10)
    ax.set_yticks([])
    ax.set_xlabel("IC50 Value")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.6), ncol=3, frameon=False)
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("🔬 Powered by PyTorch • Streamlit • SHAP • GDSC")
