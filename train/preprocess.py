"""
preprocess.py
Loads the raw Telco Customer Churn CSV and returns a cleaned DataFrame
ready for feature engineering.
"""
import pandas as pd
import numpy as np


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop the unique identifier - not predictive
    df = df.drop(columns=["customerID"])

    # TotalCharges is stored as text and has blank strings for customers
    # with tenure == 0 (i.e. brand-new customers who haven't been billed yet).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # SeniorCitizen is already 0/1 but stored as int - leave as is
    # Standardize Yes/No target to 1/0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Feature engineering: average monthly spend so far (guards against tenure=0)
    df["AvgMonthlySpend"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
    )

    return df


if __name__ == "__main__":
    df = load_and_clean("../data/telco.csv")
    print(df.shape)
    print(df.isna().sum().sum(), "missing values remaining")
    print(df["Churn"].value_counts(normalize=True))
    df.to_csv("../data/telco_clean.csv", index=False)
    print("Saved cleaned data to data/telco_clean.csv")
