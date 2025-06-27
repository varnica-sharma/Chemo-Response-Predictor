import streamlit as st
import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import IC50Net

# Load model and preprocessing tools
torch.serialization.add_safe_globals({"IC50Net": IC50Net})
model = torch.load("models/ic50_model.pt", weights_only=False)
model.eval()

scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/onehot_encoder.pkl")
label_enc_drug = joblib.load("models/drug_encoder.pkl")
label_enc_cell = joblib.load("models/cell_encoder.pkl")

# Streamlit Page Config
st.set_page_config(
    page_title="Personalized Chemotherapy Response Predictor",
    page_icon="💊",
    layout="wide"
)

# Title
st.title("🔬 Prediction using Genomic & Drug Metadata (GDSC)")

# Sidebar Inputs
st.sidebar.header("🧬 Input Patient & Drug Info")
drug_name = st.sidebar.selectbox("Drug Name", label_enc_drug.classes_)
cell_line = st.sidebar.selectbox("Cell Line", label_enc_cell.classes_)
tissue = st.sidebar.selectbox("Tissue Type", ['breast', 'lung', 'skin', 'bone', 'blood'])
tcga = st.sidebar.selectbox("TCGA Classification", ['brca', 'luad', 'skcm', 'lusc', 'gbm'])

# Predict button
if st.sidebar.button("🧠 Predict IC50"):

    # --- Input Encoding ---
    drug_enc = label_enc_drug.transform([drug_name])[0]
    cell_enc = label_enc_cell.transform([cell_line])[0]
    z_score = 0.0
    max_conc = 10.0
    z_scaled, max_scaled = scaler.transform([[z_score, max_conc]])[0]

    meta_df = pd.DataFrame([[tissue.lower(), tcga.lower()]], columns=["Tissue", "TCGA_Classification"])
    meta_onehot = encoder.transform(meta_df)

    x = np.hstack([[cell_enc, drug_enc, z_scaled, max_scaled], meta_onehot[0]]).astype(np.float32)
    x_tensor = torch.tensor([x], dtype=torch.float32)

    # --- Prediction ---
    with torch.no_grad():
        pred = model(x_tensor).item()

    # --- Result Box ---
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

    # --- Interpretation Message ---
    st.markdown("### 🧠 Interpretation of Prediction")
    if pred < 4:
        interpretation = "**high sensitivity** to the selected drug. This means the drug is likely to be effective at lower concentrations."
    elif pred < 6:
        interpretation = "**moderate sensitivity**. The drug may work, but higher doses might be needed."
    else:
        interpretation = "**low sensitivity or resistance**. The drug may be less effective at typical doses."

    st.markdown(f"""
    <div style='
        background-color: #1E392A;
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-size: 1rem;
    '>
    🔷 The predicted IC50 indicates {interpretation}
    </div>
    """, unsafe_allow_html=True)

    # --- Visualization ---
    def plot_ic50_interpretation(predicted_ic50):
        fig, ax = plt.subplots(figsize=(8, 1.8))
        ax.axvspan(0, 4, color='lightgreen', label='Sensitive')
        ax.axvspan(4, 6, color='khaki', label='Moderate')
        ax.axvspan(6, 10, color='lightcoral', label='Resistant')
        ax.axvline(predicted_ic50, color='black', linestyle='--', linewidth=2)
        ax.text(predicted_ic50 + 0.1, 1.02, f'Predicted = {round(predicted_ic50, 2)}',
                transform=ax.get_xaxis_transform(), color='black')

        ax.text(2, 0.5, 'Sensitive', ha='center', va='center', fontsize=12)
        ax.text(5, 0.5, 'Moderate', ha='center', va='center', fontsize=12)
        ax.text(8, 0.5, 'Resistant', ha='center', va='center', fontsize=12)

        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1)
        ax.axis('off')
        st.pyplot(fig)

    plot_ic50_interpretation(pred)

# Footer
st.markdown("---")
st.markdown("🧪 Powered by PyTorch • Streamlit • GDSC")
