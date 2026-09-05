import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide"
)

# ============================================================
# LOAD SAVED MODELS AND OBJECTS
# ============================================================

MODEL_DIR = "saved_models"


@st.cache_resource
def load_models():

    models = {
        "Decision Tree": joblib.load(
            os.path.join(MODEL_DIR, "decision_tree.pkl")
        ),
        "Random Forest": joblib.load(
            os.path.join(MODEL_DIR, "random_forest.pkl")
        ),
        "XGBoost": joblib.load(
            os.path.join(MODEL_DIR, "xgboost.pkl")
        )
    }

    scaler = joblib.load(
        os.path.join(MODEL_DIR, "scaler.pkl")
    )

    encoder = joblib.load(
        os.path.join(MODEL_DIR, "label_encoder.pkl")
    )

    feature_names = joblib.load(
        os.path.join(MODEL_DIR, "feature_names.pkl")
    )

    return models, scaler, encoder, feature_names


models, scaler, encoder, feature_names = load_models()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Control Panel")

selected_model = st.sidebar.selectbox(
    "Select ML Model",
    ["Decision Tree", "Random Forest", "XGBoost"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Model Information")
st.sidebar.info(
    f"**Active Model:** {selected_model}\n\n"
    f"**Features:** {len(feature_names)}\n\n"
    f"**Feature Names:**\n"
    + "\n".join([f"- {f}" for f in feature_names])
)

# ============================================================
# MAIN PAGE HEADER
# ============================================================

st.title("🏭 Predictive Maintenance System")
st.markdown(
    "Enter machine sensor readings below to predict whether "
    "the machine will fail and estimate maintenance costs."
)

st.markdown("---")

# ============================================================
# INPUT FORM — SENSOR READINGS
# ============================================================

st.subheader("🔧 Machine Sensor Inputs")

col1, col2, col3 = st.columns(3)

with col1:

    machine_type = st.selectbox(
        "Machine Type",
        options=["L", "M", "H"],
        help="L = Low, M = Medium, H = High quality variant"
    )

    air_temp = st.number_input(
        "Air Temperature [K]",
        min_value=290.0,
        max_value=310.0,
        value=298.1,
        step=0.1,
        format="%.1f"
    )

with col2:

    process_temp = st.number_input(
        "Process Temperature [K]",
        min_value=300.0,
        max_value=315.0,
        value=308.6,
        step=0.1,
        format="%.1f"
    )

    rotational_speed = st.number_input(
        "Rotational Speed [rpm]",
        min_value=1000,
        max_value=3000,
        value=1551,
        step=1
    )

with col3:

    torque = st.number_input(
        "Torque [Nm]",
        min_value=3.0,
        max_value=80.0,
        value=42.8,
        step=0.1,
        format="%.1f"
    )

    tool_wear = st.number_input(
        "Tool Wear [min]",
        min_value=0,
        max_value=250,
        value=0,
        step=1
    )

# ============================================================
# INPUT FORM — FAILURE MODE INDICATORS
# ============================================================

st.subheader("🛑 Failure Mode Indicators")

fm_col1, fm_col2, fm_col3, fm_col4, fm_col5 = st.columns(5)

with fm_col1:
    twf = st.selectbox("Tool Wear Failure (TWF)", [0, 1])

with fm_col2:
    hdf = st.selectbox("Heat Dissipation Failure (HDF)", [0, 1])

with fm_col3:
    pwf = st.selectbox("Power Failure (PWF)", [0, 1])

with fm_col4:
    osf = st.selectbox("Overstrain Failure (OSF)", [0, 1])

with fm_col5:
    rnf = st.selectbox("Random Failure (RNF)", [0, 1])

st.markdown("---")

# ============================================================
# MAINTENANCE COST SETTINGS (EXPANDABLE)
# ============================================================

with st.expander("💰 Maintenance Cost Settings (Click to customise)"):

    cost_col1, cost_col2 = st.columns(2)

    with cost_col1:
        inspection_cost = st.number_input(
            "Inspection Cost (£)",
            value=100,
            step=10
        )
        failure_cost = st.number_input(
            "Failure Cost (£)",
            value=5000,
            step=100
        )

    with cost_col2:
        downtime_cost = st.number_input(
            "Downtime Cost (£)",
            value=3000,
            step=100
        )
        preventive_cost = st.number_input(
            "Preventive Maintenance Cost (£)",
            value=300,
            step=10
        )

# ============================================================
# PREDICTION
# ============================================================

if st.button("🔍 Predict Machine Failure", type="primary", use_container_width=True):

    # --- Encode machine type ---
    type_encoded = encoder.transform([machine_type])[0]

    # --- Original column names (as the scaler and DT/RF were trained) ---
    original_columns = [
        'Type',
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]',
        'TWF',
        'HDF',
        'PWF',
        'OSF',
        'RNF'
    ]

    # --- Build input dataframe ---
    raw_input = pd.DataFrame([[
        type_encoded,
        air_temp,
        process_temp,
        rotational_speed,
        torque,
        tool_wear,
        twf,
        hdf,
        pwf,
        osf,
        rnf
    ]], columns=original_columns)

    # --- Scale the input (uses original column names) ---
    scaled_input = scaler.transform(raw_input)
    scaled_df = pd.DataFrame(scaled_input, columns=original_columns)

    # --- Get the selected model ---
    model = models[selected_model]

    # --- Prepare model input based on model type ---
    # XGBoost was trained with bracket-cleaned column names
    # Decision Tree and Random Forest keep original names
    if selected_model == "XGBoost":
        model_input = scaled_df.copy()
        model_input.columns = (
            model_input.columns.astype(str)
            .str.replace('[', '', regex=False)
            .str.replace(']', '', regex=False)
            .str.replace('<', '', regex=False)
        )
    else:
        model_input = scaled_df.copy()

    # --- Predict ---
    prediction = model.predict(model_input)[0]

    # --- Predict probability ---
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(model_input)[0]
        failure_prob = probabilities[1]
        no_failure_prob = probabilities[0]
    else:
        failure_prob = float(prediction)
        no_failure_prob = 1 - failure_prob

    # ========================================================
    # RESULTS
    # ========================================================

    st.markdown("---")
    st.subheader("📋 Prediction Results")

    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        if prediction == 1:
            st.error("⚠️ **MACHINE FAILURE PREDICTED**")
        else:
            st.success("✅ **NO FAILURE PREDICTED**")

    with res_col2:
        st.metric(
            label="Failure Probability",
            value=f"{failure_prob * 100:.2f}%"
        )

    with res_col3:
        st.metric(
            label="Model Used",
            value=selected_model
        )

    # ========================================================
    # MAINTENANCE COST ESTIMATION
    # ========================================================

    st.markdown("---")
    st.subheader("💰 Maintenance Cost Estimation")

    if prediction == 1:
        estimated_cost = preventive_cost
        avoided_cost = (failure_cost + downtime_cost) - preventive_cost
        action = "Schedule Preventive Maintenance"
        st.warning(f"**Recommended Action:** {action}")

        mc1, mc2, mc3 = st.columns(3)

        with mc1:
            st.metric(
                "Preventive Maintenance Cost",
                f"£{preventive_cost:,}"
            )

        with mc2:
            st.metric(
                "Potential Failure Cost (if ignored)",
                f"£{failure_cost + downtime_cost:,}"
            )

        with mc3:
            st.metric(
                "Cost Saved by Acting Now",
                f"£{avoided_cost:,}"
            )

    else:
        st.info(
            "**Recommended Action:** Continue Normal Operation. "
            "No immediate maintenance required."
        )
        st.metric(
            "Estimated Current Cost",
            "£0"
        )

    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    st.markdown("---")
    st.subheader("🧠 Model Explanation (SHAP)")

    if selected_model == "XGBoost":

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(model_input)

        fig, ax = plt.subplots(figsize=(10, 4))

        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=model_input.iloc[0].values,
                feature_names=list(model_input.columns)
            ),
            show=False
        )

        st.pyplot(fig)
        plt.close()

    elif selected_model == "Random Forest":

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(model_input)

        fig, ax = plt.subplots(figsize=(10, 4))

        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        ev = (
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        )

        shap.waterfall_plot(
            shap.Explanation(
                values=sv[0],
                base_values=ev,
                data=model_input.iloc[0].values,
                feature_names=list(model_input.columns)
            ),
            show=False
        )

        st.pyplot(fig)
        plt.close()

    else:
        st.info(
            "SHAP waterfall plot is available for "
            "Random Forest and XGBoost models. "
            "Decision Tree rules are shown below instead."
        )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    st.markdown("---")
    st.subheader("📊 Feature Importance")

    if hasattr(model, "feature_importances_"):

        importance_df = pd.DataFrame({
            "Feature": list(model_input.columns),
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.barh(
            importance_df["Feature"],
            importance_df["Importance"],
            color="#4CAF50"
        )

        ax.set_xlabel("Importance")
        ax.set_title(f"Feature Importance — {selected_model}")

        st.pyplot(fig)
        plt.close()

    # ========================================================
    # INPUT SUMMARY TABLE
    # ========================================================

    st.markdown("---")
    st.subheader("📝 Input Summary")

    summary = pd.DataFrame({
        "Parameter": [
            "Machine Type",
            "Air Temperature [K]",
            "Process Temperature [K]",
            "Rotational Speed [rpm]",
            "Torque [Nm]",
            "Tool Wear [min]",
            "Tool Wear Failure (TWF)",
            "Heat Dissipation Failure (HDF)",
            "Power Failure (PWF)",
            "Overstrain Failure (OSF)",
            "Random Failure (RNF)"
        ],
        "Value": [
            machine_type,
            air_temp,
            process_temp,
            rotational_speed,
            torque,
            tool_wear,
            twf,
            hdf,
            pwf,
            osf,
            rnf
        ]
    })

    st.table(summary)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:grey;'>"
    "Predictive Maintenance System | "
    "Built with Streamlit | "
    "AI4I 2020 Dataset"
    "</p>",
    unsafe_allow_html=True
)