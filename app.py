import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Dubai Real Estate AI", layout="wide")

@st.cache_data
def load_data():
    # Load your downloaded CSV (keep first 50k rows for speed)
    df = pd.read_csv('dubai_transactions.csv', nrows=50000)
    # Basic Cleaning
    df = df[df['property_type_en'] == 'Unit']
    return df[['area_name_en', 'procedure_area', 'actual_worth']].dropna()

df = load_data()

st.title("🏙️ Dubai Property Price Predictor")
st.sidebar.header("Filter Property")

# Encoding area names
area = st.sidebar.selectbox("Select Area", df['area_name_en'].unique())
sqft = st.sidebar.slider("Area (SqFt)", 400, 5000, 10000)

# Quick Model Training (In a real project, save the model as a .pkl)
if st.button("Predict Price"):
    # Simple processing for demo
    df_encoded = pd.get_dummies(df, columns=['area_name_en'])
    X = df_encoded.drop('actual_worth', axis=1)
    y = df_encoded['actual_worth']
    
    model = RandomForestRegressor(n_estimators=10) # Fast for demo
    model.fit(X[:5000], y[:5000])
    
    # Create input vector
    input_data = pd.DataFrame(columns=X.columns)
    input_data.loc[0] = 0
    input_data.at[0, 'procedure_area'] = sqft
    if f'area_name_en_{area}' in input_data.columns:
        input_data.at[0, f'area_name_en_{area}'] = 1
        
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Price: AED {prediction:,.0f}")
