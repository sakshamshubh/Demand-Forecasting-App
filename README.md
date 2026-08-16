# Demand Forecasting using XGBoost

An end-to-end machine learning project that predicts product demand using an XGBoost Regressor, selected after comparing six candidate models, with an interactive web app for real-time demand prediction, deployed live on Streamlit Community Cloud.

**Live App:** https://dmd-frcst-ml.streamlit.app/

---

## Overview

Accurately forecasting product demand helps businesses optimize inventory, pricing, and promotional strategy. This project builds a regression model that predicts demand based on pricing, discounting, promotional activity, competitor pricing, and product category, using a retail dataset of **76,000 records** across multiple stores and product categories.

---

## Objective

- Predict product demand using key pricing and behavioral features
- Compare multiple regression algorithms before committing to one
- Tune the top candidates to minimize prediction error using hyperparameter optimization
- Deploy an interactive tool that predicts demand in real time from user-provided inputs

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | XGBoost (XGBRegressor), Scikit-learn (Linear/Ridge/Lasso, Decision Tree, Random Forest, OneHotEncoder, StandardScaler, RandomizedSearchCV) |
| Model Persistence | Pickle |
| Web App / Deployment | Streamlit, Streamlit Community Cloud |

---

## Project Workflow

**1. Feature Selection**
Selected 6 features relevant to demand prediction:
`Price`, `Discount`, `Inventory Level`, `Promotion`, `Competitor Pricing`, `Category`

**2. Encoding**
- Identified `Category` as the only categorical feature
- Applied `OneHotEncoder` (with `drop="first"`) instead of label encoding, since Category is nominal with no natural order, label encoding would have falsely implied an ordinal relationship between categories

**3. Scaling**
- Applied `StandardScaler` only to the linear models (Linear Regression, Ridge, Lasso), since unscaled features distort their coefficients
- Tree-based models (Decision Tree, Random Forest, XGBoost) were left unscaled, as scaling has no effect on how they split

**4. Train-Test Split**
- Split the data 80/20 into training and test sets

**5. Model Comparison**
- Compared six regressors on the held-out test set: Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, and XGBoost
- Random Forest and XGBoost clearly outperformed the rest and were carried forward for tuning

**6. Hyperparameter Tuning**
- Used `RandomizedSearchCV` (25 iterations, 3-fold cross-validation) on both Random Forest and XGBoost
- Random Forest search space: `n_estimators`, `max_depth`, `min_samples_split`, `max_features`
- XGBoost search space: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`
- Optimized for lowest RMSE

**7. Final Model Selection**

| Model (tuned) | Test MAE | Test RMSE | Test R² |
|---|---|---|---|
| Random Forest | 26.85 | 35.40 | 0.432 |
| **XGBoost (selected)** | 27.23 | 35.65 | 0.425 |

Random Forest scored marginally better, but the difference is under 1% on RMSE. The tuned Random Forest also saves to a file well over 100 MB due to its depth and number of trees, while XGBoost's boosted trees save to under 1 MB. Since the accuracy gap is negligible but the file size gap is large, XGBoost was chosen as the final model, it's better suited for deployment alongside a Streamlit app in a GitHub repo.

**8. Model Deployment**
- Saved the trained model and the fitted `ColumnTransformer` preprocessor using `pickle`
- Built an interactive Streamlit app that takes live input (price, discount, inventory, promotion, competitor pricing, category) and predicts demand
- Deployed publicly via Streamlit Community Cloud

---

## Feature Importance

| Feature | Importance |
|---|---|
| Category | 0.636 |
| Promotion | 0.263 |
| Price | 0.046 |
| Discount | 0.026 |
| Competitor Pricing | 0.019 |
| Inventory Level | 0.010 |

**Category** is the strongest driver of demand, followed by **Promotion**, together accounting for the large majority of the model's predictive signal. Pricing-related features (Price, Discount, Competitor Pricing) play a smaller but meaningful role, while Inventory Level has minimal influence on predicted demand.

---

## Repository Structure

```
├── app.py                        # Streamlit app
├── xgboost_demand_model.pkl      # Trained XGBoost model
├── preprocessor.pkl              # Fitted OneHotEncoder (ColumnTransformer)
├── requirements.txt              # Python dependencies
├── runtime.txt                   # Python version for deployment
├── analysis.ipynb                # Exploratory data analysis
├── machine_learning.ipynb        # Model comparison, tuning & final selection
└── README.md
```
