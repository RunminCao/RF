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
st.set_page_config(page_title="Postoperative Complication Predictor", layout="wide")
st.title("Postoperative Early Complication Prediction Tool with SHAP Explanation")
st.markdown("Please input the following **6** clinical parameters:")

# Collect user inputs
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
            selected_label = st.selectbox(
                label=f"{feature}",
                options=props["options"],
                help="Surgical procedure type"
            )
            feature_values[feature] = props["mapping"][selected_label]

# Create input DataFrame with correct feature order
input_df = pd.DataFrame([[feature_values[f] for f in feature_names]], columns=feature_names)

# Prediction and SHAP explanation
if st.button("Run Prediction", type="primary"):
    with st.spinner("Calculating..."):
        try:
            # Predict complication probability
            proba = model.predict_proba(input_df)[0]
            risk_prob = proba[1]
            prob_percent = risk_prob * 100

            # Risk stratification
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

            # Display results
            st.subheader("Prediction Results")
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.markdown(f"### Complication Risk Level: <span style='color:{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)
            with col_res2:
                st.metric("Probability of Complication", f"{prob_percent:.2f}%")

            # SHAP Force Plot
            st.subheader("Prediction Interpretation (SHAP Force Plot)")
            st.markdown("The figure shows the contribution of each feature to the prediction: **red** increases risk, **blue** decreases risk.")

            # Extract model components
            if hasattr(model, "named_steps") and "rf" in model.named_steps:
                rf_model = model.named_steps["rf"]
                preprocessor = model.named_steps.get("preprocessor", None)
            else:
                rf_model = model
                preprocessor = None

            # Preprocess input data
            if preprocessor is not None:
                X_input = preprocessor.transform(input_df)
                if hasattr(X_input, "toarray"):
                    X_input = X_input.toarray()
            else:
                X_input = input_df.values

            # Compute SHAP values
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(X_input)

            # Extract SHAP values for positive class (complication)
            if isinstance(shap_values, list):
                shap_values_class1 = shap_values[1][0]
            else:
                shap_values_class1 = shap_values[0, :, 1]

            # Clean feature names
            if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
                raw_names = preprocessor.get_feature_names_out()
                clean_names = [re.sub(r'^(num|cat)_+', '', n).lstrip('_') for n in raw_names]
            else:
                clean_names = feature_names

            # Base value
            expected_value = explainer.expected_value
            base_value = expected_value[1] if isinstance(expected_value, list) else expected_value

            # Generate SHAP force plot
            shap.initjs()
            fig = plt.figure(figsize=(14, 4))
            shap.force_plot(
                base_value,
                shap_values_class1,
                X_input[0],
                feature_names=clean_names,
                matplotlib=True,
                show=False
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Global feature importance
            with st.expander("Global Feature Importance (Mean |SHAP Value|)"):
                st.info("Full global SHAP analysis can be included during model training.")

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
