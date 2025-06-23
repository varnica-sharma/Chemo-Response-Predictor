import streamlit as st
import torch
import joblib
import numpy as np
import pandas as pd
import shap

from model import IC50Net

torch.serialization.add_safe_globals({"IC50Net": IC50Net})
model = torch.load("models/ic50_model.pt", weights_only=False)
model.eval()
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/onehot_encoder.pkl")
label_enc_drug = joblib.load("models/drug_encoder.pkl")
label_enc_cell = joblib.load("models/cell_encoder.pkl")

# Load drug_id ↔ drug_name mapping (you should maintain this as a CSV or dictionary)
drug_mapping = pd.read_csv("models/drug_id_name_map.csv")  # Assumes two columns: Drug_ID, Drug_Name
drug_id_to_name = dict(zip(drug_mapping.Drug_ID, drug_mapping.Drug_Name))
drug_name_to_id = dict(zip(drug_mapping.Drug_Name, drug_mapping.Drug_ID))

st.set_page_config(
    page_title="Personalized Chemotherapy Response Predictor",
    page_icon="💊",
    layout="wide"
)

st.markdown("""
    <style>
        .main { background-color: #0e1117; }
        .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🧬 Input Patient & Drug Info")

drug_name = st.sidebar.selectbox("Drug Name", list(drug_name_to_id.keys()))
cell_id = st.sidebar.selectbox("Cell Line", label_enc_cell.classes_)
tissue = st.sidebar.selectbox("Tissue Type", ['breast', 'lung', 'skin', 'bone', 'blood'])
tcga = st.sidebar.selectbox("TCGA Classification", ['brca', 'luad', 'skcm', 'lusc', 'gbm'])

if st.sidebar.button("🔮 Predict IC50"):
    drug_id = drug_name_to_id[drug_name]
    drug_enc = label_enc_drug.transform([drug_id])[0]
    cell_enc = label_enc_cell.transform([cell_id])[0]

    # Use default Z_score and Max_Conc as zeros (or values you trained with)
    z_score = 0.0
    max_conc = 10.0
    z_scaled, max_scaled = scaler.transform([[z_score, max_conc]])[0]

    meta_df = pd.DataFrame([[tissue.lower(), tcga.lower()]], columns=["Tissue", "TCGA_Classification"])
    meta_onehot = encoder.transform(meta_df)

    x = np.hstack([[cell_enc, drug_enc, z_scaled, max_scaled], meta_onehot[0]]).astype(np.float32)
    x_tensor = torch.tensor([x], dtype=torch.float32)

    with torch.no_grad():
        pred = model(x_tensor).item()

    # Main Section
    st.markdown("""
    <h2 style='text-align: center;'>🎯 Predicted Drug Sensitivity</h2>
    <div style='text-align: center;'>
        <p style='font-size: 24px; color: white;'>💡 Predicted IC50: <strong style='color:#ABEBC6'>{:.4f}</strong></p>
    </div>
    """.format(pred), unsafe_allow_html=True)

    # SHAP Explainability
    st.markdown("""
    <h3 style='margin-top: 3rem;'>🧠 What Influenced This Prediction?</h3>
    <p>Here are the top 5 most influential features in the model's decision:</p>
    """, unsafe_allow_html=True)

    def model_predict(x):
        with torch.no_grad():
            return model(torch.tensor(x, dtype=torch.float32)).numpy()

    explainer = shap.Explainer(model_predict, np.array([x]))
    shap_values = explainer(np.array([x]))

    feature_names = ["cell_enc", "drug_enc", "Z_score", "Max_Conc"] + list(
        encoder.get_feature_names_out(["Tissue", "TCGA_Classification"])
    )

    shap.summary_plot(
        shap_values.values,
        features=np.array([x]),
        feature_names=feature_names,
        max_display=5
    )

# Footer
st.markdown("""
---
<div style='text-align: center;'>
    <sub>Powered by PyTorch • Streamlit • SHAP • GDSC</sub>
</div>
""", unsafe_allow_html=True)
