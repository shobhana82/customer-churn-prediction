"""
app.py
Streamlit web app for the Customer Churn Prediction project.
Run locally with: streamlit run app.py
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px


# ----------------------------- Page configuration -----------------------------

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide"
)


# ----------------------------- Load model + metadata -----------------------------

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_artifacts():
    model_dir = BASE_DIR / "model"

    pipe = joblib.load(model_dir / "churn_pipeline.joblib")

    with open(model_dir / "results.json") as f:
        results = json.load(f)

    with open(model_dir / "feature_importance.json") as f:
        importances = json.load(f)

    return pipe, results, importances


pipe, results, importances = load_artifacts()

best_model_name = results["best_model"]
best_metrics = results["results"][best_model_name]


# ----------------------------- Header -----------------------------

st.title("📉 Customer Churn Predictor")

st.caption(
    "Predicts the probability that a telecom customer will churn, trained on the "
    "IBM Telco Customer Churn dataset (7,043 customers). "
    f"Best model: **{best_model_name.replace('_', ' ').title()}** "
    f"(ROC-AUC: {best_metrics['roc_auc']}, Recall: {best_metrics['recall']})"
)


tab_predict, tab_insights, tab_about = st.tabs(
    ["🔮 Predict", "📊 Model Insights", "ℹ️ About"]
)


# ----------------------------- Predict tab -----------------------------

with tab_predict:

    st.subheader("Enter customer details")

    col1, col2, col3 = st.columns(3)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        senior = st.selectbox(
            "Senior Citizen",
            ["No", "Yes"]
        )

        partner = st.selectbox(
            "Has Partner",
            ["No", "Yes"]
        )

        dependents = st.selectbox(
            "Has Dependents",
            ["No", "Yes"]
        )

        tenure = st.slider(
            "Tenure (months)",
            0,
            72,
            12
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )


    with col2:

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["Fiber optic", "DSL", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"]
        )


    with col3:

        tech_support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            0.0,
            200.0,
            70.0,
            step=1.0
        )

        total_charges = st.number_input(
            "Total Charges ($)",
            0.0,
            10000.0,
            float(monthly_charges * tenure),
            step=10.0
        )


    if st.button(
        "Predict Churn Risk",
        type="primary",
        use_container_width=True
    ):

        avg_monthly_spend = (
            total_charges / tenure
            if tenure > 0
            else monthly_charges
        )

        input_df = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "AvgMonthlySpend": avg_monthly_spend,
        }])


        proba = pipe.predict_proba(input_df)[0, 1]

        pred = (
            "Will Churn"
            if proba >= 0.5
            else "Will Stay"
        )


        risk_level = (
            "🟢 Low"
            if proba < 0.3
            else (
                "🟡 Medium"
                if proba < 0.6
                else "🔴 High"
            )
        )


        res_col1, res_col2 = st.columns([1, 2])


        with res_col1:

            st.metric(
                "Churn Probability",
                f"{proba * 100:.1f}%"
            )

            st.metric(
                "Prediction",
                pred
            )

            st.metric(
                "Risk Level",
                risk_level
            )


        with res_col2:

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    title={"text": "Churn Risk %"},
                    gauge={
                        "axis": {
                            "range": [0, 100]
                        },
                        "bar": {
                            "color": (
                                "darkred"
                                if proba >= 0.5
                                else "darkgreen"
                            )
                        },
                        "steps": [
                            {
                                "range": [0, 30],
                                "color": "#d4f7d4"
                            },
                            {
                                "range": [30, 60],
                                "color": "#fff3cd"
                            },
                            {
                                "range": [60, 100],
                                "color": "#f8d7da"
                            },
                        ],
                    },
                )
            )

            fig.update_layout(
                height=280,
                margin=dict(
                    l=20,
                    r=20,
                    t=40,
                    b=10
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        if proba >= 0.5:

            st.warning(
                "This customer profile looks similar to past churners. "
                "Common retention levers: move to a longer contract, add "
                "security/support add-ons, or switch payment method away "
                "from electronic check."
            )

        else:

            st.success(
                "This customer profile looks stable based on historical patterns."
            )


# ----------------------------- Model Insights tab -----------------------------

with tab_insights:

    st.subheader(
        "What drives churn? (Top 15 features)"
    )

    imp_df = pd.DataFrame(importances)

    fig = px.bar(
        imp_df.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Reds",
    )

    fig.update_layout(
        height=520,
        showlegend=False,
        yaxis_title="",
        xaxis_title="Importance"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.subheader("Model comparison")

    results_df = pd.DataFrame(
        results["results"]
    ).T

    results_df.index.name = "model"

    st.dataframe(
        results_df.style.highlight_max(
            axis=0,
            color="#d4f7d4"
        ),
        use_container_width=True
    )


# ----------------------------- About tab -----------------------------

with tab_about:

    st.markdown(
        """
        **Dataset:** IBM Telco Customer Churn — 7,043 customers, 21 features, ~26.5% churn rate.

        **Pipeline:** data cleaning → feature engineering (average monthly spend) →
        one-hot encoding + scaling → trained and compared Logistic Regression,
        Random Forest, and XGBoost → selected the best model by ROC-AUC.

        **Why this matters:** churn prediction lets a business flag at-risk customers
        *before* they leave, so retention efforts (discounts, contract offers,
        proactive support) can be targeted instead of blanket campaigns.

        **Built with:** Python, Pandas, Scikit-learn, XGBoost, Streamlit, Plotly.

        **Source code:** [GitHub repository link here]
        """
    )
