import streamlit as st
import pickle
import numpy as np

# Load your trained model
model = pickle.load(open('wine_quality_model.pkl', 'rb'))

# Glowing MUBASIR ANWAR
st.markdown("""
    <h1 style='text-align: center; color: #6A1B9A;'>🍷 Wine Quality Prediction</h1>
    <h3 style='text-align: center; color: #FFD700; text-shadow: 0 0 5px #FFD700, 0 0 10px #FFD700, 0 0 15px #FFD700;'>By MUBASIR ANWAR</h3>
    """, unsafe_allow_html=True)

st.write("## Enter the wine characteristics below:")

# Input fields (Now using Sliders with default values)
fixed_acidity = st.slider('Fixed Acidity', 4.0, 16.0, 7.4)
volatile_acidity = st.slider('Volatile Acidity', 0.1, 1.5, 0.7)
citric_acid = st.slider('Citric Acid', 0.0, 1.0, 0.0)
residual_sugar = st.slider('Residual Sugar', 0.5, 15.0, 1.9)
chlorides = st.slider('Chlorides', 0.01, 0.2, 0.076)
free_sulfur_dioxide = st.slider('Free Sulfur Dioxide', 1, 70, 11)
total_sulfur_dioxide = st.slider('Total Sulfur Dioxide', 6, 300, 34)
density = st.slider('Density', 0.9900, 1.0050, 0.9978)
pH = st.slider('pH', 2.5, 4.5, 3.51)
alcohol = st.slider('Alcohol', 8.0, 15.0, 9.4)

# Predict button
if st.button('Predict Quality'):
    # Create input array
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                            chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                            pH, alcohol]])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    st.success(f"🎯 The predicted wine quality is: {prediction}")

# Footer
st.caption("Made with ❤️ by Mubasir Anwar")
