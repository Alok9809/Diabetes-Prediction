import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn.metrics
sklearn.metrics._scorer._passthrough_scorer = None

# Load the pre-trained classifier
try:
    with open('Diabetes.pkl', 'rb') as file:
        classifier = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'Diabetes.pkl' not found. Please ensure the file is in the same directory as this script.")
    st.stop()

def predict():
    st.sidebar.header('Diabetes Prediction')
    st.title('Diabetes Prediction (For Females Above 21 Years of Age)')
    st.markdown("""
    This application predicts diabetes using diagnostic measurements.  
    The data originates from the National Institute of Diabetes and Digestive and Kidney Diseases.  
    **Note:** The predictions are not medical advice. Consult a professional for health-related concerns.
    """)

    # Input fields
    name = st.text_input("Name:", key="name")
    pregnancy = st.number_input("Number of times pregnant:", min_value=0, step=1, key="pregnancy")
    glucose = st.number_input("Plasma Glucose Concentration:", min_value=0.0, key="glucose")
    bp = st.number_input("Diastolic Blood Pressure (mm Hg):", min_value=0.0, key="bp")
    skin = st.number_input("Triceps Skin Fold Thickness (mm):", min_value=0.0, key="skin")
    insulin = st.number_input("2-Hour Serum Insulin (mu U/ml):", min_value=0.0, key="insulin")
    bmi = st.number_input("Body Mass Index (weight in kg/(height in m)^2):", min_value=0.0, key="bmi")
    dpf = st.number_input("Diabetes Pedigree Function:", min_value=0.0, key="dpf")
    age = st.number_input("Age (years):", min_value=21, step=1, key="age")

    submit = st.button('Predict')

    if submit:
        # Ensure all fields are filled before prediction
        if not name.strip():
            st.warning("Please enter your name.")
        else:
            try:
                # Make prediction
                features = np.array([[pregnancy, glucose, bp, skin, insulin, bmi, dpf, age]])
                prediction = classifier.predict(features)[0]

                if prediction == 0:
                    st.success(f"Congratulations, {name}! You are not diabetic.")
                else:
                    st.error(f"{name}, unfortunately, you might be diabetic. Please consult a healthcare professional.")
                    st.markdown("""
                    **Suggestions for diabetes prevention:**  
                    - Maintain a healthy weight  
                    - Exercise regularly  
                    - Eat a balanced diet  
                    - [Read more](https://www.mayoclinic.org/diseases-conditions/type-2-diabetes/in-depth/diabetes-prevention/art-20047639)
                    """)
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

def main():
    st.markdown("<h1 style='text-align: center;'>Diabetes Prediction App</h1>", unsafe_allow_html=True)
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Select Activity", ("About", "Predict Diabetes"))

    if choice == "About":
        st.markdown("""
        ### About this App  
        This application predicts whether an individual has diabetes based on several health parameters.  
        Built with **Streamlit**, this app uses a pre-trained machine learning model.  
        """)
    elif choice == "Predict Diabetes":
        predict()

if __name__ == '__main__':
    main()
