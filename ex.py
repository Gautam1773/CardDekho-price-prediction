import pickle
import pandas as pd
import streamlit as st
import numpy as np

# Page Configuration
st.set_page_config(page_title="CarDekho Price Prediction", page_icon="🚗", layout="wide")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\Gautam\OneDrive\Desktop\vs\.venv\filtered_data.csv")

df = load_data()

# Sidebar Navigation

page = st.sidebar.selectbox("Select a Page", ["CarDekho-Price Prediction", "User Guide"])

# Page 1: CarDekho Price Prediction
if page == "CarDekho-Price Prediction":
    st.header(':blue[CarDekho Price Prediction 🚗]')
    st.write("Use the form below to input car details and predict its price.")

    # Input Form
    col1, col2 = st.columns(2)

    with col1:
        # User Inputs
        Brand = st.selectbox("Brand", options=df['Brand'].unique())
        Ft = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'LPG', 'CNG', 'Electric'])
        Bt = st.selectbox("Body Type", [
            'Hatchback', 'SUV', 'Sedan', 'MUV', 'Coupe', 'Minivans',
            'Convertibles', 'Hybrids', 'Wagon', 'Pickup Trucks'
        ])
        
        # Dynamically filter models
        filtered_models = df[
            (df['Brand'] == Brand) & 
            (df['body type'] == Bt) & 
            (df['Fuel type'] == Ft)
        ]['model'].unique()
        if len(filtered_models) == 0:
            filtered_models = ["Not Available"]
        Model = st.selectbox("Model", options=filtered_models)
        
        Tr = st.selectbox("Transmission", ['Manual', 'Automatic'])
        Owner = st.selectbox("Owner", [1, 2, 3, 4, 5])
        Model_year = st.selectbox("Model Year", options=sorted(df['modelYear'].unique()))
        IV = st.selectbox("Insurance Validity", [
            'Third Party insurance', 'Comprehensive', 'Third Party',
            'Zero Dep', '2', '1', 'Not Available'
        ])
        Km = st.slider("Kilometers Driven", min_value=100, max_value=100000, step=1000)
        ML = st.number_input("Mileage (km/l)", min_value=5, max_value=50, step=1)
        
        # Dynamically filter seats
        filtered_seats = df[df['body type'] == Bt]['Seats'].unique()
        if len(filtered_seats) == 0:
            filtered_seats = [0]
        seats = st.selectbox("Seats", options=sorted(filtered_seats))
        
        color = st.selectbox("Color", df['Color'].unique())
        city = st.selectbox("City", options=df['City'].unique())

    with col2:
        # Prediction Button
        Submit = st.button("Predict")
        
        if Submit:
            try:
                # Load the trained pipeline
                with open(r'C:\Users\Gautam\OneDrive\Desktop\vs\.venv\pipeline.pkl', 'rb') as files:
                    pipeline = pickle.load(files)
                
                # Prepare input data for prediction
                new_df = pd.DataFrame({
                    'Brand': [Brand],
                    "model": [Model],
                    'Fuel type': [Ft],
                    'body type': [Bt],
                    'transmission': [Tr],
                    'ownerNo': [Owner],
                    'modelYear': [Model_year],
                    'Insurance Validity': [IV],
                    'Kms Driven': [Km],
                    'Mileage': [ML],
                    'Seats': [seats],
                    'Color': [color],
                    'City': [city]
                })

                # Display selected details
                st.write("### Selected Car Details")
                st.write(new_df)

                # Make prediction
                prediction = pipeline.predict(new_df)
                st.success(f"The predicted price of the {Brand} ({Model}) car is: ₹{round(prediction[0], 2)} lakhs")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Page 2: User Guide
elif page == "User Guide":
    st.header('User Guide for CarDekho Price Prediction Application')
    st.write("""
    This guide explains how to use the CarDekho Price Prediction app to predict car prices based on specific features.
    
    **Steps to Use:**
    1. Navigate to the **CarDekho-Price Prediction** page using the sidebar.
    2. Fill in the car details on the left column:
       - Select the car's **Brand**, **Model**, **Fuel Type**, **Body Type**, and other attributes.
       - Adjust the **Kilometers Driven** slider to indicate the car's mileage.
    3. Click the **Predict** button to calculate the car's price.
    
    **Output:**
    - The app displays the predicted price of the car in lakhs.
    - You can view the input details in a table format for reference.
    """)

    st.write("""
    ### Example Workflow:
    - Select Brand: **BMW**
    - Select Model: **5 Series**
    - Enter details like Fuel Type, Transmission, and other car specifications.
    - Click **Predict** to view the price.
    """)

    st.markdown("For more car-related information, visit: [CarDekho](https://www.cardekho.com/)")

# Footer
st.markdown("Developed by **Gautam**")
