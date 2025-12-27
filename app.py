import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Dubai Real Estate AI", layout="wide")

st.title("🏙️ Dubai Property Price Predictor")
st.markdown("""
    This AI model predicts property transaction prices based on historical data from the **Dubai Land Department**.
    *Data Source: Open Data Portal via GitHub*
""")

@st.cache_data
def load_data():
    # NEW RELIABLE URL: This is a direct link to a hosted version of the Dubai Land Department data
    # If this link ever breaks, the app will show a helpful error message.
    url = "https://raw.githubusercontent.com/ajm-shabbir/Dubai-Real-Estate-Price-Prediction/master/data.csv"
    
    try:
        # We use error_bad_lines=False to skip any rows that might be corrupted
        df = pd.read_csv(url)
        
        # RENAME COLUMNS to match our project logic
        # Different datasets use different names. This mapping ensures our code works.
        column_mapping = {
            'location': 'area_name_en',
            'area': 'procedure_area',
            'price': 'actual_worth'
        }
        df = df.rename(columns=column_mapping)
        
        # Ensure 'procedure_area' and 'actual_worth' are numbers
        df['procedure_area'] = pd.to_numeric(df['procedure_area'], errors='coerce')
        df['actual_worth'] = pd.to_numeric(df['actual_worth'], errors='coerce')
        
        return df.dropna(subset=['area_name_en', 'procedure_area', 'actual_worth'])
        
    except Exception as e:
        st.error(f"⚠️ Primary Data Link Error: {e}")
        st.info("Attempting to load local backup...")
        # FALLBACK: If you uploaded the file to GitHub, it will look for it here
        if os.path.exists('dubai_transactions.csv'):
            return pd.read_csv('dubai_transactions.csv')
        else:
            st.warning("No local backup found. Please check your URL.")
            return pd.DataFrame()

df = load_data()

if not df.empty:
    st.success("✅ Real-time Market Data Loaded!")
    
    # Sidebar for User Inputs
    st.sidebar.header("🏠 Property Features")
    selected_area = st.sidebar.selectbox("Select Community", sorted(df['area_name_en'].unique()))
    property_size = st.sidebar.slider("Property Size (SqFt)", 
                                     int(df['procedure_area'].min()), 
                                     int(df['procedure_area'].max()), 
                                     1000)

    # Simple Layout for UI
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Market Snapshot")
        st.dataframe(df.head(10))

    with col2:
        st.subheader("AI Prediction")
        
        # --- SIMPLE ML MODEL LOGIC ---
        # 1. Prepare Data
        # For a fast demo, we'll use the top 10 most common areas to train
        top_areas = df['area_name_en'].value_counts().nlargest(20).index
        df_model = df[df['area_name_en'].isin(top_areas)].copy()
        
        le = LabelEncoder()
        df_model['area_encoded'] = le.fit_transform(df_model['area_name_en'])
        
        X = df_model[['area_encoded', 'procedure_area']]
        y = df_model['actual_worth']
        
        # 2. Train Model
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # 3. Predict
        try:
            area_code = le.transform([selected_area])[0]
            prediction = model.predict([[area_code, property_size]])[0]
            
            st.metric(label="Estimated Market Value", value=f"AED {prediction:,.0f}")
            st.info(f"The average price per SqFt in {selected_area} is AED {prediction/property_size:,.2f}")
        except:
            st.warning("Prediction for this specific area is unavailable with the current subset of data.")

else:
    st.error("Could not initialize the app. Check the data source URL.")

st.divider()
st.caption("Developed by Heramb Rajesh Jaipurkar | AI Solutions Analyst")
