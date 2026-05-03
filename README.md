# IEEE-CIS Fraud Detection

Kaggle კომპეტიცია — ელექტრონული კომერციის ტრანზაქციებში თაღლითობის გამოვლენა. შეფასება: **ROC AUC**. მონაცემები: Vesta Corporation (~590K ტრანზაქცია, 400+ feature, ~3.5% fraud rate).

DagsHub MLflow: https://dagshub.com/delibaaa/DELIBA-ML-ASSIGNMENT-2.mlflow

---

## რეპოზიტორიის სტრუქტურა

```
notebooks/
├── model_experiment_LightGBM.ipynb
├── model_experiment_XGBoost.ipynb
├── model_experiment_RandomForest.ipynb
├── model_experiment_LogisticRegression.ipynb
└── model_inference.ipynb
```

---

## Feature Engineering

**ცვლადების დამუშავება:**

- OrdinalEncoder Pipeline-ის შიგნით
- Frequency encoding: `card1`, `card2`, `addr1`, `P_emaildomain`

**NaN მნიშვნელობები:**

- რიცხვითი: constant fill `-999` (tree-based) / `median` (LR, RF)
- კატეგორიული: `'missing'` constant fill

**დამატებული feature-ები:**

- `TransactionHour`, `TransactionDay`, `TransactionWeekDay` — TransactionDT-დან
- `TransactionAmt_log`, `TransactionAmt_decimal`
- `card1_amt_mean`, `card1_amt_std`, `card1_amt_diff` — aggregations
- `card1_freq`, `addr1_freq` — frequency encoding
- `P_email_suffix` — email domain suffix

**Cleaning:**

- სვეტები >90% NaN — წაშლა
- Constant სვეტები — წაშლა
- Duplicate rows — წაშლა

---

## Feature Selection

| მიდგომა                               | შედეგი                          |
| ------------------------------------- | ------------------------------- |
| High NaN drop (>90%)                  | ~50 სვეტი წაშლილი               |
| Constant column drop                  | ~5 სვეტი წაშლილი                |
| Correlation filter (>0.95)            | ~20 სვეტი წაშლილი               |
| LightGBM/XGBoost importance threshold | საბოლოო feature set             |
| RF top-100 importance                 | RF-სთვის შეზღუდვა სიჩქარის გამო |

---

## Training

### გატესტებული მოდელები

| მოდელი              | CV AUC (5-fold) | შენიშვნა                   |
| ------------------- | --------------- | -------------------------- |
| LightGBM            | ~0.926          | საუკეთესო შედეგი           |
| XGBoost             | ~0.921          | ახლოს LightGBM-თან         |
| Random Forest       | ~0.895          | ნელი                       |
| Logistic Regression | ~0.820          | underfit — linear boundary |

### Hyperparameter ოპტიმიზაცია

ყველა მოდელისთვის: **Optuna** (30 trial, 3-fold CV შიდა ციკლი).

### Overfit / Underfit ანალიზი

- **Overfit (LightGBM):** `num_leaves=512`, `min_child_samples=1` → Train AUC ~0.999, Val AUC ~0.88, gap ~0.12
- **Underfit (LightGBM):** `num_leaves=4`, `n_estimators=10` → CV AUC ~0.72
- **Underfit (Logistic Regression):** Linear model ვერ ჭერს non-linear patterns → CV AUC ~0.82
- **საუკეთესო:** LightGBM + Optuna → CV AUC ~0.926

### საბოლოო მოდელი

LightGBM Pipeline — რეგისტრირებულია Model Registry-ში: `LightGBM_Fraud_Pipeline`

---

## MLflow Tracking

| Experiment                    | Run-ები                                                                        |
| ----------------------------- | ------------------------------------------------------------------------------ |
| `LightGBM_Training`           | Cleaning, FE, Feature_Selection, CV_Baseline, Tuning, Overfit, Underfit, Final |
| `XGBoost_Training`            | იგივე სტრუქტურა                                                                |
| `RandomForest_Training`       | იგივე სტრუქტურა                                                                |
| `LogisticRegression_Training` | + C=0.001..10 regularization runs                                              |

**დალოგილი მეტრიკები:** `cv_auc_mean`, `cv_auc_std`, `train_auc`, `val_auc`, `overfit_gap`, `final_cv_auc_mean`

**Model Registry:** `LightGBM_Fraud_Pipeline` — საბოლოო submission-ისათვის
