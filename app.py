import streamlit as st
import pandas as pd
import numpy as np
import pickle

@st.cache_resource
def load_artifacts():
    with open("xgboost_demand_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    return model, preprocessor

model, preprocessor = load_artifacts()

st.title("Demand Forecasting App")

st.divider()

st.header("Input Features")

price = st.number_input("Price", min_value = 0.0, value = 50.0)
discount = st.number_input("Discount (%)", min_value = 0, max_value = 100, value = 10)
inventory_level = st.number_input("Inventory Level", min_value = 0, value = 100)
promotion = st.selectbox("Promotion", [0,1])
competitor_pricing = st.number_input("Competitor Price", min_value = 0.0, value = 50.0)

category_options = preprocessor.named_transformers_["category"].categories_[0].tolist()
category = st.selectbox("Category", category_options)

input_data = pd.DataFrame({
    "Price" : [price],
    "Discount" : [discount],
    "Inventory Level" : [inventory_level],
    "Promotion" : [promotion],
    "Competitor Pricing" : [competitor_pricing],
    "Category" : [category]
})

input_encoded = preprocessor.transform(input_data)

st.divider()

if st.button("Predict Demand"):
    prediction = max(0, model.predict(input_encoded)[0])
    st.success(f"Predicted Demand: {int(prediction)} units")
