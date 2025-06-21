import streamlit as st
import torch
import joblib
import numpy as np
import pandas as pd

from model import IC50Net
torch.serialization.add_safe_globals({"IC50Net": IC50Net})
model = torch.load("models/ic50_model.pt", weights_only=False)
model.eval()
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/onehot_encoder.pkl")
label_enc_drug = joblib.load("models/drug_encoder.pkl")
label_enc_cell = joblib.load("models/cell_encoder.pkl")

# Title
st.title("💊 Personalized Chemotherapy Response Prediction using Genomic & Drug Metadata (GDSC)")

# Sidebar inputs
st.sidebar.header("Input Patient & Drug Info")

drug_id = st.sidebar.selectbox("Drug ID", label_enc_drug.classes_)
cell_id = st.sidebar.selectbox("Cell Line", label_enc_cell.classes_)
tissue = st.sidebar.selectbox("Tissue", ['breast', 'lung', 'skin', 'bone', 'blood'])
tcga = st.sidebar.selectbox("TCGA", ['brca', 'luad', 'skcm', 'lusc', 'gbm'])
z_score = st.sidebar.slider("Z-score", -3.0, 3.0, 0.0)
max_conc = st.sidebar.slider("Max Concentration", 0.0, 20.0, 10.0)

if st.sidebar.button("🔮 Predict IC50"):
    drug_enc = label_enc_drug.transform([drug_id])[0]
    cell_enc = label_enc_cell.transform([cell_id])[0]
    z_scaled, max_scaled = scaler.transform([[z_score, max_conc]])[0]
    meta_df = pd.DataFrame([[tissue.lower(), tcga.lower()]], columns=["Tissue", "TCGA_Classification"])
    meta_onehot = encoder.transform(meta_df)

    x = np.hstack([[cell_enc, drug_enc, z_scaled, max_scaled], meta_onehot[0]]).astype(np.float32)
    x_tensor = torch.tensor([x], dtype=torch.float32)

    with torch.no_grad():
        pred = model(x_tensor).item()

    st.success(f"🔬 Predicted IC50: **{round(pred, 4)}**")

