import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained Random Forest model using the exact new file name
try:
    model = joblib.load('model_rf.pkl')
except Exception as e:
    st.error(f"Error loading the model file: {str(e)}")

# Page configuration and title setup
st.set_page_config(page_title="Student Productivity Predictor", layout="centered")
st.title("🎓 Student Productivity & Academic Success Predictor")
st.write("Enter your daily routine metrics to predict your Productivity Score based on the updated AI model.")

st.markdown("---")

# User Inputs section matching the exact 6 features from the new Colab notebook
st.subheader("📊 Daily Routine Factors")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.number_input("Study Hours Per Day", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
    focus_score = st.slider("Focus Score (30 - 100)", min_value=30, max_value=100, value=65)
    sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0, step=0.5)

with col2:
    phone_usage = st.number_input("Phone Usage Hours", min_value=0.0, max_value=24.0, value=3.0, step=0.5)
    stress_level = st.slider("Stress Level (1 - 10)", min_value=1, max_value=10, value=5)
    attendance = st.slider("Attendance Percentage (%)", min_value=40, max_value=100, value=85)

st.markdown("---")

# Prediction handling triggered by the button
if st.button("🚀 Predict Productivity Score", use_container_width=True):
    # Construct the input array using the exact sequence of the 6 features
    user_data = np.array([[study_hours, focus_score, sleep_hours, phone_usage, stress_level, attendance]])
    
    try:
        # Generate prediction using the loaded model
        prediction = model.predict(user_data)
        predicted_score = prediction[0]
        
        # Display results and trigger celebratory balloons animation
        st.balloons()
        st.success(f"### 📈 Predicted Productivity Score: {predicted_score:.2f}%")
        
        # Provide feedback based on the predicted score threshold
        if predicted_score > 75:
            st.info("🎯 Great job! This routine indicates high productivity and strong academic focus.")
        elif predicted_score > 50:
            st.warning("⚠️ Moderate productivity. Adjusting study hours or reducing phone usage could boost performance.")
        else:
            st.error("🚨 Low productivity risk. High stress/distraction levels or low study hours detected.")
    except Exception as e:
        st.error(f"Prediction Error: {str(e)}. Please ensure the inputs match model configuration.")
