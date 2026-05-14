import streamlit as st
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# Load the trained model
try:
    model = joblib.load('rf.pkl')
except FileNotFoundError:
    st.error("Model file 'rf.pkl' not found. Please upload the model file.")
    st.stop()

# Feature names (must match the training data order)
feature_names = [
    "Operation method",
    "iPTH_T1",
    "iPTH_T2",
    "TPV",
    "BonePain",
    "P_T0"
]

# Feature input ranges and types
feature_ranges = {
    "Operation method": {
        "type": "categorical",
        "options": ["SPTX (0)", "TPTX (1)", "TPTX+AT (2)"],
        "mapping": {"SPTX (0)": 0, "TPTX (1)": 1, "TPTX+AT (2)": 2}
    },
    "iPTH_T1": {"type": "numerical", "min": 0.0, "max": 5000.0, "default": 100.0, "step": 5.0},
    "iPTH_T2": {"type": "numerical", "min": 0.0, "max": 5000.0, "default": 100.0, "step": 5.0},
    "TPV":     {"type": "numerical", "min": 0.0, "max": 10.0, "default": 1.0, "step": 0.1},
    "BonePain": {"type": "numerical", "min": 0.0, "max": 10.0, "default": 0.0, "step": 0.5},
    "P_T0":    {"type": "numerical", "min": 0.0, "max": 20.0, "default": 3.0, "step": 0.1},
}

# Page configuration
st.set_page_config(page_title="Predictors of recurrence after PTX", layout="wide")
st.title("Predictive tool for recurrence after PTX and SHAP explanation")
st.markdown("Please input the following **6** clinical parameters:")

# Input UI
feature_values = {}
col1, col2 = st.columns(2)
for i, (feature, props) in enumerate(feature_ranges.items()):
    with col1 if i % 2 == 0 else col2:
        if props["type"] == "numerical":
            feature_values[feature] = st.number_input(
                label=f"{feature}",
                min_value=float(props["min"]),
                max_value=float(props["max"]),
                value=float(props["default"]),
                step=props.get("step", 1.0),
                help=f"Range: {props['min']} - {props['max']}"
            )
        else:
            selected_label = st.selectbox(label=f"{feature}", options=props["options"])
            feature_values[feature] = props["mapping"][selected_label]

# Input DataFrame
input_df = pd.DataFrame([[feature_values[f] for f in feature_names]], columns=feature_names)

# Predict
if st.button("Run Prediction", type="primary"):
    with st.spinner("Calculating..."):
        try:
            # Prediction
            proba = model.predict_proba(input_df)[0]
            risk_prob = proba[1]
            prob_percent = risk_prob * 100

            # Risk level
            low_th = 0.33
            high_th = 0.67
            if risk_prob < low_th:
                risk_level = "🔵 Low Risk"
                risk_color = "blue"
            elif risk_prob < high_th:
                risk_level = "🟡 Moderate Risk"
                risk_color = "orange"
            else:
                risk_level = "🔴 High Risk"
                risk_color = "red"

            # Show result
            st.subheader("Prediction Results")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"### Risk Level: <span style='color:{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)
            with c2:
                st.metric("Complication Probability", f"{prob_percent:.2f}%")

            # SHAP Plot
            st.subheader("Prediction Interpretation (SHAP Force Plot)")
            st.markdown("Red features increase risk; blue features reduce risk.")

            # Get model
            if hasattr(model, "named_steps") and "rf" in model.named_steps:
                rf_model = model.named_steps["rf"]
                preprocessor = model.named_steps.get("preprocessor", None)
            else:
                rf_model = model
                preprocessor = None

            # Preprocess
            if preprocessor is not None:
                X_input = preprocessor.transform(input_df)
                if hasattr(X_input, "toarray"):
                    X_input = X_input.toarray()
            else:
                X_input = input_df.values

            # SHAP values
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(X_input)

            # Get positive class SHAP
            if isinstance(shap_values, list):
                sv = shap_values[1][0]
            else:
                sv = shap_values[0, :, 1]

            # Base value
            ev = explainer.expected_value
            base_val = ev[1] if isinstance(ev, (list, np.ndarray)) else ev

            # Clean feature names
            if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
                names = preprocessor.get_feature_names_out()
                clean_names = [re.sub(r'^(num|cat)_+', '', n).strip('_') for n in names]
            else:
                clean_names = feature_names

            # -------------------------- 关键修复：删除 initjs() --------------------------
            fig = plt.figure(figsize=(14, 4))
            shap.force_plot(
                base_val,
                sv,
                X_input[0],
                feature_names=clean_names,
                matplotlib=True,
                show=False
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            with st.expander("Global Feature Importance"):
                st.info("Mean absolute SHAP values represent overall feature impact.")

        except Exception as e:
            st.error(f"Error: {str(e)}")
