# Customer Churn Prediction

Predict which telecom customers are likely to stop using a service.

## What it does
1. **EDA** – churn rate, churn by contract type, tenure and monthly charges (saved to `outputs/eda.png`)
2. **Models** – Logistic Regression, Random Forest, XGBoost
3. **Evaluation** – Accuracy, Recall, ROC-AUC (saved to `outputs/results.csv` and `outputs/roc.png`)

## Quick start
```bash
pip install -r requirements.txt
python churn.py
```

## Dataset
Download the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
from Kaggle and save it as `data/telco_churn.csv`.

If no file is found, the script runs on synthetic data so you can try it immediately.

## Project structure
```
churn-prediction/
├── churn.py            # EDA, training, evaluation
├── requirements.txt
├── data/               # put the CSV here
└── outputs/            # plots and results are saved here
```

## Notes
- Recall matters most for churn: it measures how many real churners we catch.
- Possible next steps: handle class imbalance (`class_weight="balanced"`), tune hyperparameters, add feature importance.
