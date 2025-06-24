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

        # SHAP Explainability
    st.markdown("### 🧠 What Influenced This Prediction?")
    st.write("Here are the top 5 most influential features in the model's decision:")

    # SHAP wrapper
    def model_predict(x):
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32)
            preds = model(x_tensor).numpy()
        return preds

    # Background for explainer
    background = np.tile(x, (100, 1))
    explainer = shap.Explainer(model_predict, background)
    shap_values = explainer(np.array([x]))

    # Feature names
    feature_names = ["cell_enc", "drug_enc", "Z_score", "Max_Conc"] + list(encoder.get_feature_names_out(["Tissue", "TCGA_Classification"]))

    # Manual bar plot
    shap_vals = shap_values.values[0]
    top_idx = np.argsort(np.abs(shap_vals))[-5:][::-1]
    top_features = [feature_names[i] for i in top_idx]
    top_shap_vals = shap_vals[top_idx]

    fig, ax = plt.subplots()
    ax.barh(top_features[::-1], top_shap_vals[::-1], color='salmon')
    ax.set_xlabel("SHAP Value (Impact on IC50)")
    ax.set_title("Top 5 Influential Features")
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("🔬 Powered by PyTorch • Streamlit • SHAP • GDSC")
