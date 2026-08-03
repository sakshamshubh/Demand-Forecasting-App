# Demand Forecasting using XGBoost

An end-to-end machine learning project that predicts product demand using an XGBoost Regressor, with an interactive web app for real-time demand prediction — deployed live on Streamlit Community Cloud.

**Live App:** https://dmd-frcst-ml.streamlit.app/

---

## Overview

Accurately forecasting product demand helps businesses optimize inventory, pricing, and promotional strategy. This project builds a regression model that predicts demand based on pricing, discounting, promotional activity, competitor pricing, and product category, using a retail dataset of **76,000 records** across multiple stores and product categories.

---

## Objective

- Predict product demand using key pricing and behavioral features
- Tune the model to minimize prediction error using hyperparameter optimization
- Deploy an interactive tool that predicts demand in real time from user-provided inputs

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | XGBoost (XGBRegressor), Scikit-learn (LabelEncoder, RandomizedSearchCV) |
| Model Persistence | Pickle |
| Web App / Deployment | Streamlit, Streamlit Community Cloud |

---

## Project Workflow

**1. Feature Selection**
Selected 6 features relevant to demand prediction:
`Price`, `Discount`, `Inventory Level`, `Promotion`, `Competitor Pricing`, `Category`

**2. Encoding**
- Identified `Category` as the only categorical feature
- Applied `LabelEncoder` to convert it into a numeric format suitable for the model

**3. Train-Test Split**
- Split the data 80/20 into training and test sets

**4. Model Selection**
- Used `XGBRegressor` — a tree-based model, so feature scaling was not required

**5. Hyperparameter Tuning**
- Used `RandomizedSearchCV` (25 iterations, 3-fold cross-validation) to search across:
  `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`
- Optimized for lowest Mean Absolute Error

**6. Evaluation**
- Achieved a Root Mean Squared Error (RMSE) of **~35.8 units** on the test set
- Reviewed feature importances to understand key demand drivers

**7. Model Deployment**
- Saved the trained model and label encoders using `pickle`
- Built an interactive Streamlit app that takes live input (price, discount, inventory, promotion, competitor pricing, category) and predicts demand
- Deployed publicly via Streamlit Community Cloud

---

## Feature Importance

| Feature | Importance |
|---|---|
| Promotion | 0.607 |
| Category | 0.273 |
| Price | 0.065 |
| Competitor Pricing | 0.021 |
| Discount | 0.017 |
| Inventory Level | 0.016 |

**Promotion** was by far the strongest driver of demand, followed by **Category** — together accounting for the majority of the model's predictive signal. Pricing-related features (Price, Competitor Pricing, Discount) played a smaller but meaningful role, while Inventory Level had minimal influence on predicted demand.

---

## Repository Structure

```
├── app.py                        # Streamlit app
├── xgboost_demand_model.pkl      # Trained XGBoost model
├── label_encoders.pkl            # Fitted label encoders
├── requirements.txt              # Python dependencies
├── runtime.txt                   # Python version for deployment
├── analysis.ipynb                # Exploratory data analysis
├── machine_learning.ipynb        # Feature engineering, tuning & model training
└── README.md
```
