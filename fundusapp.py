import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model

model = load_model(r"E:\fundus\fundus.h5")

class_labels = ['Normal', 'Diabetic Retinopathy', 'Glaucoma', 'Cataract']

def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_fundus_image(model, image):
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)
    predictions_dict = {class_labels[i]: prediction[0][i] * 100 for i in range(len(class_labels))}
    return predictions_dict

st.title("Fundus Image Classification App")


image1 = st.file_uploader("Choose the first fundus image", type=["jpg", "jpeg", "png"])
image2 = st.file_uploader("Choose the second fundus image", type=["jpg", "jpeg", "png"])


if st.button("Predict"):

    if image1 and image2:

        st.image([image1, image2], caption=['Image 1', 'Image 2'], use_column_width=True)


        predictions1 = predict_fundus_image(model, image1)


        predictions2 = predict_fundus_image(model, image2)


        st.write("Predictions for Image 1:")
        for class_name, confidence in predictions1.items():
            st.write(f"{class_name}: {confidence:.2f}%")


        st.write("Predictions for Image 2:")
        for class_name, confidence in predictions2.items():
            st.write(f"{class_name}: {confidence:.2f}%")
