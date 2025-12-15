import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

@st.cache_resource
def load_model():
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, 'cifar10_model.h5')
    model = tf.keras.models.load_model(model_path, compile=False)
    return model


model = load_model()

class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']


st.title("Image Classification Web App")
st.write("Upload an image, and the Deep Learning model will predict what it is.")
st.write("(Supported classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck)")

file = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])

def import_and_predict(image_data, model):
    size = (32, 32)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img = img / 255.0
    img_reshape = np.reshape(img, (1, 32, 32, 3))
    prediction = model.predict(img_reshape)
    return prediction

if file is not None:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    if st.button("Predict"):
        predictions = import_and_predict(image, model)
        score = tf.nn.softmax(predictions[0])
        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)
        
        st.success(f"Prediction: **{predicted_class}**")
        st.info(f"Confidence: {confidence:.2f}%")