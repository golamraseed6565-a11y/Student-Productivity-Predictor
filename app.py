import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. Saved model aur scaler ko load karna
model = joblib.load('productivity_model.pkl')
scaler = joblib.load('scaler.pkl')

# 2. Page ki settings aur Title
st.set_page_config(page_title="Student Productivity Predictor", layout="centered")
st.title("🎓 Student Productivity & Academic Success Predictor")
st.write("Enter daily routine metrics to predict the Productivity Score based on AI Models.")

st.markdown("---")

# 3. User Inputs (Dataset ke factors ke mutabiq sliders aur boxes)
st.subheader("📊 Daily Routine Factors")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.number_input("Study Hours Per Day", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
    sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
    focus_score = st.slider("Focus Score (1 - 100)", min_value=1, max_value=100, value=60)
    attendance = st.slider("Attendance Percentage (%)", min_value=0, max_value=100, value=85)

with col2:
    phone_usage = st.number_input("Phone Usage Hours", min_value=0.0, max_value=24.0, value=3.0, step=0.5)
    social_media = st.number_input("Social Media Hours", min_value=0.0, max_value=24.0, value=1.5, step=0.5)
    stress_level = st.slider("Stress Level (1 - 10)", min_value=1, max_value=10, value=5)
    assignments = st.number_input("Assignments Completed", min_value=0, max_value=50, value=10, step=1)

st.markdown("---")

# 4. Prediction Button
if st.button("🚀 Predict Productivity Score", use_container_width=True):
    # Inputs ka array banana (Sahi sequence mein)
    user_data = np.array([[study_hours, sleep_hours, phone_usage, social_media, stress_level, focus_score, attendance, assignments]])
    
    # Inputs ko scale karna (Scaler file ke zariye)
    user_data_scaled = scaler.transform(user_data)
    
    # Model se prediction lena
    prediction = model.predict(user_data_scaled)
    predicted_score = prediction[0]
    
    # Result show karna aur balloons udana
    st.balloons()
    st.success(f"### 📈 Predicted Productivity Score: {predicted_score:.2f}%")
    
    # Score ke mutabiq choti si advice show karna
    if predicted_score > 75:
        st.info("🎯 Great job! This routine indicates high productivity and strong academic focus.")
    elif predicted_score > 50:
        st.warning("⚠️ Moderate productivity. Reducing phone/social media usage could boost performance.")
    else:
        st.error("🚨 Low productivity risk. High distraction levels or low study hours detected.")
