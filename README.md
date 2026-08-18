# 📉 Customer Churn Prediction

Predicts whether a telecom customer is likely to churn, and serves the model
through an interactive web app — built end-to-end: data cleaning → feature
engineering → model comparison → deployment.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**🔗 Live app:** _[add your Streamlit Cloud link here after deploying]_
**📓 Dataset:** [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d)

<!--
📸 Add a screenshot or GIF of the app here once deployed:
![App screenshot](assets/app_screenshot.png)
-->

## Table of Contents

- [Problem](#problem)
- [Dataset](#dataset)
- [Approach](#approach)
- [Results](#results)
- [Project Structure](#project-structure)
- [Run it Locally](#run-it-locally)
- [Deploy for Free](#deploy-it-for-free-streamlit-community-cloud)
- [Tech Stack](#tech-stack)
- [Future Improvements](#future-improvements)
- [Author](#author)

## Problem

Telecom companies lose recurring revenue every time a customer cancels.
Retention campaigns are expensive to run for the whole customer base, so
the business needs a way to flag **which** customers are actually at risk
of leaving — so retention effort (discounts, contract offers, proactive
support) is targeted instead of blanket.

## Dataset

[IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) —
**7,043 customers**, **21 features** (demographics, account info, services
subscribed), **~26.5% churn rate**.

## Approach

1. **Data cleaning** — fixed `TotalCharges` (stored as text, blank for
   brand-new customers with tenure = 0), dropped the customer ID, mapped
   the target to 0/1.
2. **Feature engineering** — added `AvgMonthlySpend` (total charges ÷
   tenure) as a signal beyond the raw columns.
3. **Modeling** — trained and compared three models on an 80/20 stratified
   split, using class weighting to handle the ~27% churn imbalance.
4. **Model selection** — picked the winner by **ROC-AUC and recall**, not
   raw accuracy (see [why](#results) below).
5. **Feature importance** — pulled out what actually drives churn.
6. **Deployment** — wrapped the winning model in a Streamlit app with a
   live prediction form, gauge chart, and model-insights dashboard.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.738 | 0.504 | 0.781 | 0.613 | 0.842 |
| **Random Forest ✅ (selected)** | **0.757** | **0.530** | **0.759** | **0.624** | **0.842** |
| XGBoost | 0.745 | 0.514 | 0.762 | 0.614 | 0.840 |

> **Why not just optimize for accuracy?** Only ~26.5% of customers churn,
> so a model that predicts "no churn" for everyone would still score ~73%
> accuracy while being completely useless. Random Forest was selected on
> ROC-AUC, with recall weighted heavily — missing an actual churner costs
> the business more than a false alarm does.

**Top churn drivers** (feature importance): month-to-month contracts, low
tenure, no online security / tech support add-ons, fiber optic internet,
and paying by electronic check.

## Project Structure

```
churn_project/
├── data/
│   └── telco.csv              # raw dataset
├── train/
│   ├── preprocess.py          # cleaning
│   ├── train_model.py         # trains & compares models, saves the best one
│   └── feature_importance.py  # extracts top drivers of churn
├── app/
│   ├── app.py                 # Streamlit app
│   ├── requirements.txt
│   └── model/                 # trained pipeline + metrics (generated)
└── README.md
```

## Run it Locally

```bash
# 1. clone the repo
git clone https://github.com/<your-username>/customer-churn-prediction.git
cd customer-churn-prediction

# 2. install dependencies
pip install -r app/requirements.txt

# 3. (optional) retrain the model — a trained one is already included in app/model/
cd train
python preprocess.py
python train_model.py
python feature_importance.py

# 4. run the app
cd ../app
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Deploy it for Free (Streamlit Community Cloud)

1. Push this whole folder to a public GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select your repo → set the main file path to `app/app.py`.
4. Click **Deploy**. You'll get a public URL like `https://your-app.streamlit.app`.
5. Add that link to the top of this README, your resume, and LinkedIn.

## Tech Stack

**Language:** Python
**Data & ML:** Pandas, NumPy, Scikit-learn, XGBoost
**App & Visualization:** Streamlit, Plotly
**Model persistence:** joblib

## Future Improvements

- Hyperparameter tuning with GridSearchCV / Optuna
- SHAP values for per-prediction explainability (why *this* customer is
  flagged, not just global feature importance)
- Model monitoring for data drift if retrained on newer data
- CI pipeline to auto-retrain and re-deploy on new data

## Author

**Shobhana**
[GitHub](https://github.com/shobhana82) · [LinkedIn](#) · [Portfolio](#)

---

⭐ If you found this project useful, consider starring the repo!
