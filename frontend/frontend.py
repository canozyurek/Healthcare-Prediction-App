import streamlit as st
import requests
import pandas as pd


def run():
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
        [
            "Welcome",
            "Stroke Prediction",
            "Hypertension Prediction",
            "Glucose Prediction",
            "BMI Prediction",
            "Hypertension & Glucose Prediction",
            "Hypertension & BMI Prediction",
            "Glucose & BMI Prediction",
            "Hypertension, Glucose & BMi Prediciton",
        ]
    )

    with tab1:
        st.header("Health Issues Prediction System")
        st.markdown("An application to predict a number of health problems.")
        st.markdown("This application is not always accurate.")
    with tab2:
        st.header("Stroke Prediction")
        st.markdown("Please enter the required information:")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age")
        hypertension = st.selectbox("Hypertension", ["Yes", "No"])
        heart_disease = st.selectbox("Heart Disease", ["Yes", "No"])
        ever_married = st.selectbox("Ever Married", ["Yes", "No"])
        work_type = st.selectbox(
            "Work Type", ["Private", "Self-employed", "Govt_job", "Never_worked"]
        )
        Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
        avg_glucose_level = st.number_input("Average Glucose Level")
        bmi = st.number_input("BMI")
        smoking_status = st.selectbox(
            "Smoking Status", ["formerly smoked", "never smoked", "smokes"]
        )

        data = {
            "gender": gender,
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "ever_married": ever_married,
            "work_type": work_type,
            "Residence_type": Residence_type,
            "avg_glucose_level": avg_glucose_level,
            "bmi": bmi,
            "smoking_status": smoking_status,
        }

        if st.button("Predict"):
            response = requests.post("http://127.0.0.1:8000/predict-stroke", json=data)
            prediction = response.text
            st.success(f"The prediction from model: {prediction}")
    with tab3:
        st.header("Hypertension Prediction")
        st.markdown("Please enter the required information:")
        gender = st.selectbox("Gender", ["Male", "Female"], key=1)
        age = st.number_input("Age", key=2)
        heart_disease = st.selectbox("Heart Disease", ["Yes", "No"], key=4)
        ever_married = st.selectbox("Ever Married", ["Yes", "No"], key=5)
        work_type = st.selectbox(
            "Work Type", ["Private", "Self-employed", "Govt_job", "Never_worked"], key=6
        )
        Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"], key=7)
        avg_glucose_level = st.number_input("Average Glucose Level", key=8)
        bmi = st.number_input("BMI", key=9)
        smoking_status = st.selectbox(
            "Smoking Status", ["formerly smoked", "never smoked", "smokes"], key=10
        )

        data = {
            "gender": str(gender),
            "age": int(age),
            "heart_disease": str(heart_disease),
            "ever_married": str(ever_married),
            "work_type": str(work_type),
            "Residence_type": str(Residence_type),
            "avg_glucose_level": float(avg_glucose_level),
            "bmi": float(bmi),
            "smoking_status": str(smoking_status),
        }
        if st.button("Predict", key=11):
            response = requests.post(
                "http://127.0.0.1:8000/predict-hypertension", json=data
            )
            prediction = response.text
            st.success(f"The prediction from model: {prediction}")

    with tab4:
        st.header("Glucose Prediction")
        st.markdown("Please enter the required information:")
        gender = st.selectbox("Gender", ["Male", "Female"], key=12)
        age = st.number_input("Age", key=22)
        hypertension = st.selectbox("Hypertension", ["Yes", "No"], key=32)
        heart_disease = st.selectbox("Heart Disease", ["Yes", "No"], key=42)
        ever_married = st.selectbox("Ever Married", ["Yes", "No"], key=52)
        work_type = st.selectbox(
            "Work Type",
            ["Private", "Self-employed", "Govt_job", "Never_worked"],
            key=62,
        )
        Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"], key=72)
        bmi = st.number_input("BMI", key=92)
        smoking_status = st.selectbox(
            "Smoking Status", ["formerly smoked", "never smoked", "smokes"], key=102
        )

        data = {
            "gender": str(gender),
            "age": int(age),
            "hypertension": hypertension,
            "heart_disease": str(heart_disease),
            "ever_married": str(ever_married),
            "work_type": str(work_type),
            "Residence_type": str(Residence_type),
            "bmi": float(bmi),
            "smoking_status": str(smoking_status),
        }

        if st.button("Predict", key=112):
            response = requests.post("http://127.0.0.1:8000/predict-glucose", json=data)
            prediction = response.text
            st.success(f"Predicted average glucose level: {prediction}")

    with tab5:
        st.header("BMI Prediction")
        st.markdown("Please enter the required information:")
        gender = st.selectbox("Gender", ["Male", "Female"], key=13)
        age = st.number_input("Age", key=23)
        hypertension = st.selectbox("Hypertension", ["Yes", "No"], key=33)
        heart_disease = st.selectbox("Heart Disease", ["Yes", "No"], key=43)
        ever_married = st.selectbox("Ever Married", ["Yes", "No"], key=53)
        work_type = st.selectbox(
            "Work Type",
            ["Private", "Self-employed", "Govt_job", "Never_worked"],
            key=63,
        )
        Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"], key=73)
        avg_glucose_level = st.number_input("Average Glucose Level", key=83)
        smoking_status = st.selectbox(
            "Smoking Status", ["formerly smoked", "never smoked", "smokes"], key=103
        )

        data = {
            "gender": str(gender),
            "age": int(age),
            "hypertension": hypertension,
            "heart_disease": str(heart_disease),
            "ever_married": str(ever_married),
            "work_type": str(work_type),
            "Residence_type": str(Residence_type),
            "avg_glucose_level": float(avg_glucose_level),
            "smoking_status": str(smoking_status),
        }

        if st.button("Predict", key=113):
            response = requests.post("http://127.0.0.1:8000/predict-bmi", json=data)
            prediction = response.text
            st.success(f"Predicted BMI: {prediction}")

    with tab6:
        st.header("Hypertension and Glucose Prediction")
        st.markdown("Please enter the required information:")
        gender = st.selectbox("Gender", ["Male", "Female"], key=14)
        age = st.number_input("Age", key=24)
        heart_disease = st.selectbox("Heart Disease", ["Yes", "No"], key=44)
        ever_married = st.selectbox("Ever Married", ["Yes", "No"], key=54)
        work_type = st.selectbox(
            "Work Type",
            ["Private", "Self-employed", "Govt_job", "Never_worked"],
            key=64,
        )
        Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"], key=74)
        bmi = st.number_input("BMI", key=84)
        smoking_status = st.selectbox(
            "Smoking Status", ["formerly smoked", "never smoked", "smokes"], key=104
        )

        data = {
            "gender": str(gender),
            "age": int(age),
            "heart_disease": str(heart_disease),
            "ever_married": str(ever_married),
            "work_type": str(work_type),
            "Residence_type": str(Residence_type),
            "bmi": float(bmi),
            "smoking_status": str(smoking_status),
        }

        if st.button("Predict", key=114):
            response = requests.post("http://127.0.0.1:8000/predict-hypglu", json=data)
            prediction = response.text
            prediction = prediction.split(",")
            st.success(
                f"Predicted hypertension risk: {prediction[0].replace('[', '')}\
            Predicted glucose level: {prediction[1].replace(']', '')}"
            )

    with tab7:
        st.header("Hypertension and BMI Prediction")
        st.markdown("Please enter the required information:")
        gender = st.selectbox("Gender", ["Male", "Female"], key=15)
        age = st.number_input("Age", key=25)
        heart_disease = st.selectbox("Heart Disease", ["Yes", "No"], key=45)
        ever_married = st.selectbox("Ever Married", ["Yes", "No"], key=55)
        work_type = st.selectbox(
            "Work Type",
            ["Private", "Self-employed", "Govt_job", "Never_worked"],
            key=65,
        )
        Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"], key=75)
        avg_glucose_level = st.number_input("Average Glucose Level", key=85)
        smoking_status = st.selectbox(
            "Smoking Status", ["formerly smoked", "never smoked", "smokes"], key=105
        )

        data = {
            "gender": str(gender),
            "age": int(age),
            "heart_disease": str(heart_disease),
            "ever_married": str(ever_married),
            "work_type": str(work_type),
            "Residence_type": str(Residence_type),
            "avg_glucose_level": float(avg_glucose_level),
            "smoking_status": str(smoking_status),
        }

        if st.button("Predict", key=115):
            response = requests.post("http://127.0.0.1:8000/predict-hypbmi", json=data)
            prediction = response.text
            prediction = prediction.split(",")
            st.success(
                f"Predicted hypertension risk: {prediction[0].replace('[', '')}\
            Predicted BMI level: {prediction[1].replace(']', '')}"
            )

    with tab8:
        st.header("Glucose and BMI Prediction")
        st.markdown("Please enter the required information:")
        gender = st.selectbox("Gender", ["Male", "Female"], key=16)
        age = st.number_input("Age", key=26)
        heart_disease = st.selectbox("Heart Disease", ["Yes", "No"], key=46)
        ever_married = st.selectbox("Ever Married", ["Yes", "No"], key=56)
        work_type = st.selectbox(
            "Work Type",
            ["Private", "Self-employed", "Govt_job", "Never_worked"],
            key=66,
        )
        Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"], key=76)
        smoking_status = st.selectbox(
            "Smoking Status", ["formerly smoked", "never smoked", "smokes"], key=106
        )
        hypertension = st.selectbox("Hypertension", ["Yes", "No"], key=36)

        data = {
            "gender": str(gender),
            "age": int(age),
            "hypertension": hypertension,
            "heart_disease": str(heart_disease),
            "ever_married": str(ever_married),
            "work_type": str(work_type),
            "Residence_type": str(Residence_type),
            "smoking_status": str(smoking_status),
        }

        if st.button("Predict", key=116):
            response = requests.post("http://127.0.0.1:8000/predict-glubmi", json=data)
            prediction = response.text
            st.success(f"Predicted : {prediction}")

    with tab9:
        st.header("Hypertension, Glucose and BMI Prediction")
        st.markdown("Please enter the required information:")
        gender = st.selectbox("Gender", ["Male", "Female"], key=17)
        age = st.number_input("Age", key=27)
        heart_disease = st.selectbox("Heart Disease", ["Yes", "No"], key=47)
        ever_married = st.selectbox("Ever Married", ["Yes", "No"], key=57)
        work_type = st.selectbox(
            "Work Type",
            ["Private", "Self-employed", "Govt_job", "Never_worked"],
            key=67,
        )
        Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"], key=77)
        smoking_status = st.selectbox(
            "Smoking Status", ["formerly smoked", "never smoked", "smokes"], key=107
        )

        data = {
            "gender": str(gender),
            "age": int(age),
            "heart_disease": str(heart_disease),
            "ever_married": str(ever_married),
            "work_type": str(work_type),
            "Residence_type": str(Residence_type),
            "smoking_status": str(smoking_status),
        }

        if st.button("Predict", key=117):
            response = requests.post(
                "http://127.0.0.1:8000/predict-hypglubmi", json=data
            )
            prediction = response.text
            prediction = prediction.split(",")
            st.success(
                f"Predicted hypertension risk: {prediction[0].replace('[', '')}\n\
            Predicted glucose level: {prediction[1].replace(']', '')}\
            Predicted BMI level: {prediction[2].replace(']', '')}"
            )


if __name__ == "__main__":
    run()
