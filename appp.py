import streamlit as st
import pickle
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
models = {
    "Gradient Boosting": joblib.load('gb_model.pkl'),
    "XGBoost": joblib.load('xgb_model.pkl')
}
scale = joblib.load('scale.pkl')

selected_model_name = st.sidebar.selectbox("Choose a Model", list(models.keys()))
selected_model = models[selected_model_name]
def user_input_parameters():
    st.sidebar.header("Patient Vitals")

    # 1. Numerical Inputs
    age = st.sidebar.slider('Age', 0,100)
    bmi = st.sidebar.number_input('BMI', 10.0, 50.0, 25.0)
    systolic_bp = st.sidebar.number_input('Systolic BP', 90, 200, 120)
    diastolic_bp = st.sidebar.number_input('Diastolic BP', 60, 120, 80)
    chol_total = st.sidebar.number_input('Cholesterol Total', 100, 300, 200)
    chol_ldl = st.sidebar.number_input('Cholesterol LDL', 50, 200, 100)
    chol_hdl = st.sidebar.number_input('Cholesterol HDL', 20, 100, 50)
    chol_tri = st.sidebar.number_input('Cholesterol Triglycerides', 50, 400, 150)
    mmse = st.sidebar.slider('MMSE Score', 0, 30, 20)
    functional_assessment = st.sidebar.slider('Functional Assessment', 0, 10, 5)
    adl = st.sidebar.slider('ADL Score', 0, 10, 5)
    sleep_quality = st.sidebar.slider('Sleep Quality (1-10)', 1, 10, 5)
    diet_quality = st.sidebar.slider('Diet Quality (1-10)', 1, 10, 5)
    physical_activity = st.sidebar.slider('Physical Activity (1-10)', 1, 10, 5)
    alcohol = st.sidebar.slider('Alcohol Consumption (1-10)', 1, 10, 5)

    # 2. Categorical/Binary Inputs
    gender = st.sidebar.selectbox('Gender (0=Male, 1=Female)', [0, 1])
    ethnicity = st.sidebar.selectbox('Ethnicity (0, 1, 2, 3)', [0, 1, 2, 3])
    education = st.sidebar.selectbox('Education Level (0, 1, 2, 3)', [0, 1, 2, 3])
    smoking = st.sidebar.selectbox('Smoking (0=No, 1=Yes)', [0, 1])
    family_history = st.sidebar.selectbox('Family History (0=No, 1=Yes)', [0, 1])
    cardio = st.sidebar.selectbox('Cardiovascular Disease (0=No, 1=Yes)', [0, 1])
    diabetes = st.sidebar.selectbox('Diabetes (0=No, 1=Yes)', [0, 1])
    depression = st.sidebar.selectbox('Depression (0=No, 1=Yes)', [0, 1])
    head_injury = st.sidebar.selectbox('Head Injury (0=No, 1=Yes)', [0, 1])
    hypertension = st.sidebar.selectbox('Hypertension (0=No, 1=Yes)', [0, 1])
    memory_complaints = st.sidebar.selectbox('Memory Complaints (0=No, 1=Yes)', [0, 1])
    behavioral_problems = st.sidebar.selectbox('Behavioral Problems (0=No, 1=Yes)', [0, 1])
    confusion = st.sidebar.selectbox('Confusion (0=No, 1=Yes)', [0, 1])
    disorientation = st.sidebar.selectbox('Disorientation (0=No, 1=Yes)', [0, 1])
    personality_changes = st.sidebar.selectbox('Personality Changes (0=No, 1=Yes)', [0, 1])
    difficulty_tasks = st.sidebar.selectbox('Difficulty Tasks (0=No, 1=Yes)', [0, 1])
    forgetfulness = st.sidebar.selectbox('Forgetfulness (0=No, 1=Yes)', [0, 1])
    data = {
        'Age': age, 'Gender': gender, 'Ethnicity': ethnicity, 'EducationLevel': education,
        'BMI': bmi, 'Smoking': smoking, 'AlcoholConsumption': alcohol, 'PhysicalActivity': physical_activity,
        'DietQuality': diet_quality, 'SleepQuality': sleep_quality, 'FamilyHistoryAlzheimers': family_history,
        'CardiovascularDisease': cardio, 'Diabetes': diabetes, 'Depression': depression,
        'HeadInjury': head_injury, 'Hypertension': hypertension, 'SystolicBP': systolic_bp,
        'DiastolicBP': diastolic_bp, 'CholesterolTotal': chol_total, 'CholesterolLDL': chol_ldl,
        'CholesterolHDL': chol_hdl, 'CholesterolTriglycerides': chol_tri, 'MMSE': mmse,
        'FunctionalAssessment': functional_assessment, 'MemoryComplaints': memory_complaints,
        'BehavioralProblems': behavioral_problems, 'ADL': adl, 'Confusion': confusion,
        'Disorientation': disorientation, 'PersonalityChanges': personality_changes,
        'DifficultyCompletingTasks': difficulty_tasks, 'Forgetfulness': forgetfulness,
    }
    features= pd.DataFrame(data,index=[0])
    num_cols = [
        'Age', 'BMI', 'MMSE', 'ADL', 'FunctionalAssessment',
        'AlcoholConsumption', 'PhysicalActivity', 'SleepQuality',
        'SystolicBP', 'DiastolicBP', 'CholesterolTotal',
        'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides'
    ]
    features[num_cols] = scale.transform(features[num_cols])
    return features
df = user_input_parameters()
if st.button('Predict'):
    prediction = selected_model.predict(df)
    prob = selected_model.predict_proba(df)
    st.subheader(f'Prediction: {"Diagnosed" if prediction[0] == 1 else "Healthy"}')
    st.write(f'Probability: {prob[0][1]:.2%}')
