
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model (pipeline)
@st.cache_resource
def load_model():
    return joblib.load("superkart_pipeline.joblib")

model = load_model()

# Streamlit UI
st.title("SuperKart Sales Prediction App")
st.write("This tool predicts the expected sales revenue for a product in a specific SuperKart store.")

st.subheader("Enter Product & Store Details:")

# -----------------------------
# Collect user input (Single Prediction)
# -----------------------------

product_id = st.text_input("Product ID", "P001")
product_weight = st.number_input("Product Weight (kg)", min_value=0.0, value=10.0)
product_sugar_content = st.selectbox("Product Sugar Content", ["Low", "Regular", "No Sugar"])
product_allocated_area = st.number_input("Allocated Display Area (0 to 1)", min_value=0.0, max_value=1.0, value=0.3)
product_type = st.text_input("Product Type", "Snack")
product_mrp = st.number_input("Product MRP", min_value=0.0, value=100.0)

store_id = st.text_input("Store ID", "S001")
store_est_year = st.number_input("Store Establishment Year", min_value=1900, max_value=2100, value=2005)
store_size = st.selectbox("Store Size", ["Low", "Medium", "High"])
store_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
store_type = st.selectbox("Store Type", ["Departmental", "Supermarket Type 1", "Supermarket Type 2", "Food Mart"])

# Convert user input into a DataFrame
input_data = pd.DataFrame([{
    "Product_Id": product_id,
    "Product_Weight": product_weight,
    "Product_Sugar_Content": product_sugar_content,
    "Product_Allocated_Area": product_allocated_area,
    "Product_Type": product_type,
    "Product_MRP": product_mrp,
    "Store_Id": store_id,
    "Store_Establishment_Year": store_est_year,
    "Store_Size": store_size,
    "Store_Location_City_Type": store_city_type,
    "Store_Type": store_type
}])

# Predict button
if st.button("Predict Sales"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Sales Revenue: ₹{prediction:.2f}")
