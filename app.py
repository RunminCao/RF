import streamlit as st
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Predictors of recurrence after PTX", layout="wide")

# Load model
try:
    model = joblib.load('rf.pkl')
except FileNotFoundError:
    st.error("Model file 'rf.pkl' not found. Please upload the model file.")
    st.stop()

# Feature names (must match training order)
feature_names = [
    "Operation method",
    "iPTH_T1",
    "iPTH_T2",
    "TPV",
    "BonePain",
    "P_T0"
]

# Feature input ranges (ALL float type to avoid Streamlit error)
feature_ranges = {
    "Operation method": {
        "type": "categorical",
        "options": ["SPTX (0)", "TPTX (1)", "TPTX+AT (2)"],
        "mapping": {"SPTX (0)": 0, "TPTX (1)": 1, "TPTX+AT (2)": 2}
    },
    "iPTH_T1": {"type": "numerical", "min": 0.0, "max": 5000.0, "default": 100.0, "step": 5.0},
    "iPTH_T2": {"type": "numerical", "min": 0.0, "max": 5000.0, "default": 100.0, "step": 5.0},
    "TPV":     {"type": "numerical", "min": 0.0, "max": 10.0, "default": 1.0, "step": 0.1},
    "BonePain":{"type": "numerical", "min": 0.0, "max": 10.0, "default": 0.0, "step": 0.5},
    "P_T0":    {"type": "numerical", "min": 0.0, "max": 20.0, "default": 3.0, "step": 0.1},
}

# UI
st.title("Predictive tool for recurrence after PTX and SHAP explanation")
st.markdown("Please input the following **6** clinical parameters:")

# Get user inputs
feature_values = {}
col1, col2 = st.columns(2)

for i, (feature, props) in enumerate(feature_ranges.items()):
    with col1 if i % 2 == 0 else col2:
        if props["type"] == "numerical":
            feature_values[feature] = st.number_input(
                label=feature,
                min_value=props["min"],
                max_value=props["max"],
                value=props["default"],
                step=props["step"]
            )
        else:
            sel = st.selectbox(label=feature, options=props["options"])
            feature_values[feature] = props["mapping"][sel]

# Create input dataframe
input_df = pd.DataFrame([[feature_values[f] for f in feature_names]], columns=feature_names)

# Run prediction
if st.button("Run Prediction", type="primary"):
    with st.spinner("Calculating..."):
        try:
            proba = model.predict_proba(input_df)[0]
            risk_prob = proba[1]
            prob_percent = risk_prob * 100

            # Risk level
            low_th = 0.33
            high_th = 0.67
            if risk_prob < low_th:
                risk_level = "🔵 Low Risk"
                color = "blue"
            elif risk_prob < high_th:
                risk_level = "🟡 Moderate Risk"
                color = "orange"
            else:
                risk_level = "🔴 High Risk"
                color = "red"

            st.subheader("Prediction Results")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"### Risk Level: <span style='color:{color}'>{risk_level}</span>", unsafe_allow_html=True)
            with c2:
                st.metric("Recurrence Probability", f"{prob_percent:.2f}%")

            # ==============================
            # SHAP Waterfall → SHOW ORIGINAL INPUT VALUES
            # ==============================
            st.subheader("Prediction Interpretation (SHAP Waterfall Plot)")
            st.markdown("Red = increases risk | Blue = decreases risk")

            # Get model
            if hasattr(model, "named_steps"):
                rf = model.named_steps["rf"]
                pre = model.named_steps.get("preprocessor", None)
            else:
                rf = model
                pre = None

            # Scaled data for SHAP calculation
            if pre:
                X_scaled = pre.transform(input_df)
                if hasattr(X_scaled, "toarray"):
                    X_scaled = X_scaled.toarray()
            else:
                X_scaled = input_df.values

            # SHAP values
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_scaled)

            if isinstance(shap_values, list):
                sv = shap_values[1][0]
            else:
                sv = shap_values[0][:, 1]

            # Base value
            ev = explainer.expected_value
            base_val = ev[1] if isinstance(ev, (list, np.ndarray)) else ev

            # ==============================
            # KEY FIX: Show ORIGINAL input values (10, not 1.36)
            # ==============================
            original_data = input_df.values[0]

            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=sv,
                    base_values=base_val,
                    data=original_data,
                    feature_names=feature_names
                ),
                show=False
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.error(f"Error: {str(e)}")
