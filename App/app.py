# Import packages
import streamlit as st
import altair as at
import plotly.express as px
import pandas as pd
import numpy as np
import joblib

# Load the model
pipeline_lr = joblib.load(open("../models/emotion_classifier.pkl", "rb"))


# Functions
def predict_emotions(text):
    results = pipeline_lr.predict([text])
    return results[0]

def predict_probability(text):
    results = pipeline_lr.predict_proba([text])
    return results

st.title("Know Your Emotions")
menu = ["Home", "About"]
choice = st.sidebar.selectbox("Menu", menu)
if choice=="Home":
    st.subheader("Home- Emotion in Text")
    with st.form(key="emotion_clf_form"):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label="Submit")
        
    if submit_text:
        col1, col2 = st.beta_columns(2)
        
        prediction = predict_emotions(raw_text)
        probability = predict_probability(raw_text)
        
        with col1:
            st.success("Original Text")
            st.write(raw_text)
            
            st.success("Prediction")
            st.write(prediction, dict_emoji[prediction])
            st.write("Confidence: ", np.max(probability))
            
        with col2:
            st.success("Prediction Probability")
            prob_df = pd.DataFrame(probability, columns=pipeline_lr.classes_)
            prob_df_clean = prob_df.T.reset_index()
            prob_df_clean.columns = ['emotions','probability']
            fig = at.Chart(prob_df_clean).mark_bar().encode(x="emotions", y="probability", color="emotions")
            st.altair_chart(fig, use_container_width=True)
            
elif choice=="About":
    st.subheader("About")