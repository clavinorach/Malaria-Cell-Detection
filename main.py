import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

def main():
    st.title("Malaria Cell Prediction")

    # Load the pre-trained VGG16 model
    model = tf.keras.models.load_model("VGG16_model.h5")

    # Display an upload file dialog and get the uploaded file
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    # Check if a file was uploaded
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image')

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make predictions
        prediction = make_prediction(model, processed_image)

        # Display the result
        if prediction == 0:
            st.write("# **IT IS NOT INFECTED WITH MALARIA PARASITE**", unsafe_allow_html=True)
            st.write("Great news! You don't have malaria.")
        else:
            st.write("# **IT IS INFECTED WITH MALARIA PARASITE**", unsafe_allow_html=True)
            st.write("Please consult a healthcare professional for further evaluation and treatment.")

def preprocess_image(image):
    # Resize the image to the required input shape of the model
    resized_image = image.resize((224, 224))

    # Convert the image to a NumPy array
    img_array = np.array(resized_image)

    # Normalize the image pixel values
    normalized_image = img_array / 255.0

    # Add an extra dimension to match the input shape of the model
    processed_image = np.expand_dims(normalized_image, axis=0)

    return processed_image

def make_prediction(model, image):
    # Make predictions using the model
    predictions = model.predict(image)

    # Get the predicted class (0: uninfected, 1: parasitized)
    predicted_class = np.argmax(predictions, axis=1)[0]

    return predicted_class

if __name__ == "__main__":
    main()
