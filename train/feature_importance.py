"""
feature_importance.py
Pulls feature importances out of the trained pipeline so we know (and can
show in the app) which factors drive churn the most.
"""
import joblib
import json
import numpy as np

pipe = joblib.load("../model/churn_pipeline.joblib")
preprocessor = pipe.named_steps["preprocessor"]
model = pipe.named_steps["model"]

num_features = preprocessor.transformers_[0][2]
cat_encoder = preprocessor.transformers_[1][1]
cat_features = cat_encoder.get_feature_names_out(preprocessor.transformers_[1][2]).tolist()

all_features = list(num_features) + cat_features
importances = model.feature_importances_

pairs = sorted(zip(all_features, importances), key=lambda x: -x[1])

top15 = pairs[:15]
for name, score in top15:
    print(f"{name:35s} {score:.4f}")

with open("../model/feature_importance.json", "w") as f:
    json.dump([{"feature": n, "importance": float(s)} for n, s in top15], f, indent=2)

print("\nSaved to model/feature_importance.json")
