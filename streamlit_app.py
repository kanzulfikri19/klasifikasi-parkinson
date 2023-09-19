import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image

# Load the trained ensemble models
majority_vote_model = joblib.load('mayoritas.pkl')

# Load the preprocessed dataset
preprocessed_data = pd.read_excel("datasetfix.xlsx")

# Add title and introduction
st.title('Klasifikasi parkinson dengan ensemble SVM')
st.write("Selamat datang!")
st.write("Pada aplikasi ini anda bisa melakukan klasifikasi parkinson melalui frekuensi fitur yang dihasilkan dari analisa vokal dengan ensemble SVM.")

# Display illustration and information about Parkinson's disease
image = Image.open('parkinson.JPEG')
st.image(image, caption='Illustration of Parkinsons Disease')

# Display the most influential features
st.write("### Fitur frekuensi suara")
most_influential_features = pd.read_excel("fitur.xlsx")
st.dataframe(most_influential_features)

# Sidebar
st.sidebar.title('Pilih sampel untuk di klasifikasikan')

# Get the index of the selected sample
selected_sample_index = st.sidebar.slider("Pilih index sampel", 0, len(preprocessed_data) - 1)

# Add a "Run" button
run_button = st.sidebar.button("Run")

# Display the selected sample's data in the main page
st.write("Sampel yang dipilih:", selected_sample_index)
st.write("Isi sampel yang dipilih:")
st.dataframe(preprocessed_data.iloc[selected_sample_index, :])

if run_button:
    # Get the features for the selected sample (excluding the first column)
    selected_features = preprocessed_data.iloc[selected_sample_index, 1:].values

    # Make prediction using the ensemble models
    majority_vote_prediction = majority_vote_model.predict([selected_features])


    # Display the predictions and probabilities in the main page
    st.write("### HASIL UJI COBA SAMPEL")
    st.write("Hasil klasifikasi:", "Parkinson's Disease" if majority_vote_prediction else "Healthy")
    
    ##st.write("Prediksi ensemble average:", "Parkinson's Disease" if ensemble_avg_prediction else "Healthy")
    ###st.write("Prediction Probabilities:")
    ##st.write(f"Class 0: {ensemble_avg_probabilities[0]:.2%}")
    ##st.write(f"Class 1: {ensemble_avg_probabilities[1]:.2%}")
    ##ensemble_avg_prediction = ensemble_avg_model.predict([selected_features])
    ##ensemble_avg_probabilities = ensemble_avg_model.predict_proba([selected_features])[0]
