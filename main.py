import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

def main():
    st.title("Malaria Cell Prediction")


    model = tf.keras.models.load_model("VGG16_model.h5")

   
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

   
    if uploaded_file is not None:
      
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image')

        processed_image = preprocess_image(image)

       
        prediction = make_prediction(model, processed_image)

       
        if prediction == 0:
            st.write("# **IT IS NOT INFECTED WITH MALARIA PARASITE**", unsafe_allow_html=True)
            st.write("Great news! You don't have malaria.")
        else:
            st.write("# **IT IS INFECTED WITH MALARIA PARASITE**", unsafe_allow_html=True)
            st.write("Please consult a healthcare professional for further evaluation and treatment.")

def preprocess_image(image):
    
    resized_image = image.resize((224, 224))

    
    img_array = np.array(resized_image)

   
    normalized_image = img_array / 255.0

    
    processed_image = np.expand_dims(normalized_image, axis=0)

    return processed_image

def make_prediction(model, image):
    
    predictions = model.predict(image)

    predicted_class = np.argmax(predictions, axis=1)[0]

    return predicted_class

if __name__ == "__main__":
    main()
