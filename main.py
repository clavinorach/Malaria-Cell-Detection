import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

def main():
    st.title("Malaria Cell Prediction Web App")
    st.write("Upload a cell image to check if it's infected with malaria parasites.")

    # Load the pre-trained model
    try:
        model = tf.keras.models.load_model("my_model.h5")
    except Exception as e:
        st.error("Error loading model: " + str(e))
        return

    # File uploader for images
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    # Prediction if file is uploaded
    if uploaded_file is not None:
        try:
            # Open and display the image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Preprocess the image for model prediction
            processed_image = preprocess_image(image)

            # Make prediction
            prediction = make_prediction(model, processed_image)

            # Display the prediction result
            if prediction == 0:
                st.success("# **IT IS NOT INFECTED WITH MALARIA PARASITE**")
                st.write("Great news! The cell appears healthy and not infected.")
            else:
                st.warning("# **IT IS INFECTED WITH MALARIA PARASITE**")
                st.write("Please consult a healthcare professional for further evaluation and treatment.")
        
        except Exception as e:
            st.error("Error processing image: " + str(e))

def preprocess_image(image):
    """
    Preprocesses the image to be suitable for model prediction.
    Resize to the input shape expected by the model and normalize pixel values.
    """
    # Resize the image to the model's input size (update size if needed)
    resized_image = image.resize((64, 64))  # Update the size based on your model input

    # Convert the image to array format
    img_array = np.array(resized_image)

    # Normalize the pixel values to the range [0, 1]
    normalized_image = img_array / 255.0

    # Expand dimensions to create a batch of size 1
    processed_image = np.expand_dims(normalized_image, axis=0)

    return processed_image

def make_prediction(model, image):
    """
    Takes the model and preprocessed image as input, returns the predicted class.
    """
    # Get prediction probabilities
    predictions = model.predict(image)

    # Get the class with the highest predicted probability
    predicted_class = np.argmax(predictions, axis=1)[0]

    return predicted_class

if __name__ == "__main__":
    main()
