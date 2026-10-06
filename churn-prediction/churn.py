"""Customer Churn Prediction: EDA + Logistic Regression, Random Forest, XGBoost."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, roc_curve

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

DATA_PATH = "data/telco_churn.csv"
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)


def make_synthetic(n=5000, seed=42):
    """Fallback data with the same columns as the Telco dataset."""
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    tenure = rng.integers(1, 73, n)
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.20])
    payment = rng.choice(["Electronic check", "Mailed check", "Credit card", "Bank transfer"], n)
    monthly = rng.normal(65, 25, n).clip(18, 120).round(2)
    logit = (-1.0 + 1.4 * (contract == "Month-to-month") - 1.2 * (contract == "Two year")
             - 0.03 * tenure + 0.015 * (monthly - 65) + 0.6 * (internet == "Fiber optic")
             + 0.4 * (payment == "Electronic check"))
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame({
        "customerID": [f"C{i:05d}" for i in range(n)],
        "tenure": tenure, "Contract": contract, "InternetService": internet,
        "PaymentMethod": payment, "MonthlyCharges": monthly,
        "TotalCharges": (monthly * tenure).round(2),
        "Churn": np.where(churn, "Yes", "No"),
    })


# ---------- 1. Load ----------
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {DATA_PATH}")
else:
    df = make_synthetic()
    print("No dataset found in data/, using synthetic data (see README to use the real one).")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
df["Churn"] = (df["Churn"] == "Yes").astype(int)
df = df.drop(columns=["customerID"])
print(f"Shape: {df.shape} | Churn rate: {df['Churn'].mean():.1%}")

# ---------- 2. EDA ----------
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.barplot(data=df, x="Contract", y="Churn", ax=ax[0]); ax[0].set_title("Churn rate by contract")
sns.histplot(data=df, x="tenure", hue="Churn", bins=30, ax=ax[1]); ax[1].set_title("Tenure vs churn")
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=ax[2]); ax[2].set_title("Monthly charges vs churn")
plt.tight_layout(); plt.savefig(f"{OUT}/eda.png", dpi=120); plt.close()

# ---------- 3. Preprocess ----------
X, y = df.drop(columns="Churn"), df["Churn"]
num_cols = X.select_dtypes("number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# ---------- 4. Models ----------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}
if XGBClassifier:
    models["XGBoost"] = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                      eval_metric="logloss", random_state=42)
else:
    print("xgboost not installed, skipping (pip install xgboost)")

# ---------- 5. Evaluate ----------
results, plt_data = [], {}
for name, model in models.items():
    pipe = Pipeline([("pre", pre), ("model", model)]).fit(X_train, y_train)
    pred, proba = pipe.predict(X_test), pipe.predict_proba(X_test)[:, 1]
    results.append({"Model": name,
                    "Accuracy": accuracy_score(y_test, pred),
                    "Recall": recall_score(y_test, pred),
                    "ROC-AUC": roc_auc_score(y_test, proba)})
    plt_data[name] = roc_curve(y_test, proba)

res = pd.DataFrame(results).round(3)
print("\n", res.to_string(index=False))
res.to_csv(f"{OUT}/results.csv", index=False)

plt.figure(figsize=(6, 5))
for name, (fpr, tpr, _) in plt_data.items():
    plt.plot(fpr, tpr, label=name)
plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves"); plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT}/roc.png", dpi=120); plt.close()
print(f"\nSaved plots and results to {OUT}/")
