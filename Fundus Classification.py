import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model

# Load the Keras model
model = load_model("C:\\Users\\sivak\\siva\\Training\\fundus_classification_model.h5")

# Define class labels
class_labels = ['Normal', 'Diabetic Retinopathy', 'Glaucoma', 'Cataract']

# Define a function to preprocess the image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Define a function to make predictions
def predict_image(model, img_array):
    prediction = model.predict(img_array)
    predictions_dict = {class_labels[i]: prediction[0][i] * 100 for i in range(len(class_labels))}
    return predictions_dict

# Streamlit app title
st.title("Fundus Image Classification App")

# File uploader for the fundus image
uploaded_image = st.file_uploader("Choose a fundus image", type=["jpg", "jpeg", "png"])

# Prediction button
if st.button("Predict"):
    if uploaded_image:
        # Display uploaded image
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

        # Preprocess and predict the uploaded image
        img_array = preprocess_image(uploaded_image)
        predictions = predict_image(model, img_array)
        st.write("Predictions:")
        for class_name, confidence in predictions.items():
            st.write(f"{class_name}: {confidence:.2f}%")
