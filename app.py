# app.py
import streamlit as st
import pandas as pd
import pickle

model_file = "iris_trained_model.p"

# --- Load the Model and Data Information ---
# Load the trained model from the pickle file
try:
    with open(model_file, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Error: '{model_file}' not found. Please run the model saving script first.")
    st.stop()

# Define the feature names and target mapping
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
target_names = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}


# --- Streamlit App Interface ---
st.title("🌺 Iris Flower Species Predictor")
st.markdown("Enter the measurements of the iris flower to predict its species.")

# --- User Input Fields ---
st.header("Flower Measurements (in cm)")

# Use st.slider for a better user experience for continuous features
sepal_length = st.slider('Sepal Length', min_value=4.0, max_value=8.0, value=5.5, step=0.1)
sepal_width = st.slider('Sepal Width', min_value=2.0, max_value=4.5, value=3.0, step=0.1)
petal_length = st.slider('Petal Length', min_value=1.0, max_value=7.0, value=4.0, step=0.1)
petal_width = st.slider('Petal Width', min_value=0.1, max_value=2.5, value=1.2, step=0.1)

# Create a DataFrame from the user input
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                          columns=feature_names)


# --- Prediction Logic ---
if st.button('**Predict Species**'):
    # Make prediction using the loaded model
    prediction_index = model.predict(input_data)[0]
    
    # Get the predicted species name
    predicted_species = target_names[prediction_index]
    
    # Get prediction probabilities (optional, but informative)
    probabilities = model.predict_proba(input_data)[0]
    
    st.subheader("✅ Prediction Result")
    st.success(f"The predicted Iris species is **{predicted_species}**.")
    
    st.write("---")
    st.subheader("Prediction Details")
    
    # Display the input data for review
    st.write("**Input Data:**")
    st.dataframe(input_data)
    
    # Display probabilities
    st.write("**Prediction Probabilities:**")
    prob_df = pd.DataFrame(probabilities.reshape(1, -1), columns=target_names.values())
    st.bar_chart(prob_df.T)