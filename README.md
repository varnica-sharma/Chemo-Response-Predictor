# 💊 Personalized Drug Response Prediction (IC50 Estimation)

This project predicts the **drug response (IC50 value)** for various cancer cell lines using **genomics-based metadata** like **tissue type, TCGA classification, drug concentration, Z-score**, and more.

Built with **PyTorch** and deployed as an interactive **Streamlit web app**, this end-to-end project is based on the **GDSC (Genomics of Drug Sensitivity in Cancer)** dataset and highlights the power of **explainable AI** in precision medicine.

---

## 🚀 Demo

🔗 [Click here to try the live app](https://chemo-response-predictor.streamlit.app)

---

## 📊 Features

- Predict **IC50** (half-maximal inhibitory concentration) for any valid combination of:
  - Drug ID
  - Cell line
  - Tissue type
  - TCGA classification
  - Max concentration
  - Z-score
- Interactive sliders and dropdowns for input
- Real-time model prediction
- SHAP-based interpretability showing top features impacting the prediction

---

## 💡 What is IC50?

> IC50 is the concentration of a drug that is required for 50% inhibition of cell viability.  
> Lower IC50 values indicate **higher sensitivity**, while higher values indicate **drug resistance**.

---

## 🔍 How It Works

1. **User selects** drug, cell line, tissue type, and TCGA classification.
2. Input features are encoded (label + one-hot + scaled).
3. The encoded vector is passed through a trained PyTorch model.
4. The predicted IC50 is shown along with an **interpretation plot**:
   - 🟩 **Sensitive:** IC50 < 4
   - 🟨 **Moderate:** 4 ≤ IC50 < 6
   - 🟥 **Resistant:** IC50 ≥ 6

---


## 📁 Dataset

- Source: [GDSC Portal](https://www.cancerrxgene.org/)
- Columns used:
  - `Drug_ID`, `Cell_Line_Name`, `Tissue`, `TCGA_Classification`, `Z_score`, `Max_Conc`, `IC50`

---

## 🧠 Model Architecture

A simple feedforward neural network (PyTorch):

- Input: Encoded features (drug, cell line, tissue, TCGA)
- Hidden Layers: 128 → 64 with ReLU + Dropout
- Output: Predicted IC50 value
- Loss: MSELoss

---

## 📈 Performance

- ✅ Final RMSE: **~1.87**
- ✅ Final R² Score: **~0.54**

> Evaluated using 80-20 train-test split on 240k+ samples.

---

## ⚙️ Tech Stack

- Python 3.11
- PyTorch
- Pandas, Scikit-learn
- Streamlit (for deployment)
- SHAP (for explainability)
