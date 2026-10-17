"""Titanic Passenger Survival Prediction System - Streamlit web app (AIC354, Assignment 2, Task 1).

The app repeats the Application Phase of the notebook (Step 8):
  take input -> build feature vector -> label-encode -> predict with the saved model.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

# ---------------------------------------------------------------
# Load the trained model saved in the notebook (Step 6.4)
# ---------------------------------------------------------------
MODEL_FILE = Path(__file__).parent / "svc_trained_model.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_FILE, "rb") as file:
        return pickle.load(file)


model = load_model()

# ---------------------------------------------------------------
# Label encoders: same value lists as notebook Step 5.1
# ---------------------------------------------------------------
pclass_values = ["First", "Second", "Third"]
gender_values = ["Male", "Female"]
sibling_values = ["Zero", "One", "Two", "Three"]
embarked_values = ["Southampton", "Cherbourg", "Queenstown"]

encoders = {
    "PClass": LabelEncoder().fit(np.ravel(pclass_values)),
    "Gender": LabelEncoder().fit(np.ravel(gender_values)),
    "Sibling": LabelEncoder().fit(np.ravel(sibling_values)),
    "Embarked": LabelEncoder().fit(np.ravel(embarked_values)),
}

# ---------------------------------------------------------------
# Page
# ---------------------------------------------------------------
st.title("🚢 Titanic Passenger Survival Prediction")
st.write(
    "Choose the passenger's details. A Support Vector Classifier, trained on "
    "100 Titanic passengers, predicts whether this passenger would have survived."
)

# Step 8.1: take input from the user
st.subheader("Passenger details")
left, right = st.columns(2)
with left:
    pclass_input = st.selectbox("Passenger class", pclass_values)
    sibling_input = st.selectbox("Siblings / spouses aboard", sibling_values)
with right:
    gender_input = st.selectbox("Gender", gender_values)
    embarked_input = st.selectbox("Port of embarkation", embarked_values)

if st.button("Predict", type="primary"):
    # Step 8.2: feature vector
    user_input = pd.DataFrame(
        {
            "PClass": [pclass_input],
            "Gender": [gender_input],
            "Sibling": [sibling_input],
            "Embarked": [embarked_input],
        }
    )

    # Step 8.3: label encoding
    unseen_data_features = user_input.copy()
    for column, encoder in encoders.items():
        unseen_data_features[column] = encoder.transform(user_input[column])

    # Step 8.5: prediction
    predicted_survival = model.predict(unseen_data_features)[0]

    if predicted_survival == 1:
        st.success("✅ Prediction: SURVIVED")
    else:
        st.error("❌ Prediction: NOT SURVIVED")

    with st.expander("How the model saw this input"):
        st.write("Your choices, converted to numbers by the Label Encoders:")
        st.dataframe(unseen_data_features, hide_index=True)

st.caption(
    "Learning project for AIC354 Machine Learning Fundamentals, COMSATS University "
    "Islamabad, Lahore Campus. Trained on a small 100-row sample, so predictions are only illustrative."
)
