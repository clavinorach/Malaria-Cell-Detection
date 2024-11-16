import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

def main():
    # Apply custom CSS for better aesthetics
    st.markdown("""
        <style>
            body {
                background-color: #f0f2f6;
            }
            .title {
                color: #4B8BBE;
                text-align: center;
                font-family: 'Helvetica';
            }
            .description {
                color: #ffffff;
                text-align: center;
                font-size: 18px;
            }
            .footer {
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background-color: #f0f2f6;
                color: #555555;
                text-align: center;
                padding: 10px 0;
                font-size: 14px;
            }
            .upload-button {
                background-color: #4B8BBE;
                color: white;
                padding: 10px 20px;
                border-radius: 5px;
                text-align: center;
                display: inline-block;
                font-size: 16px;
            }
            .prediction-success {
                color: green;
                font-size: 24px;
                text-align: center;
            }
            .prediction-warning {
                color: red;
                font-size: 24px;
                text-align: center;
            }
        </style>
        """, unsafe_allow_html=True)


    # Title and Description
    st.markdown('<h1 class="title">🦠 Malaria Cell Prediction Web App</h1>', unsafe_allow_html=True)
    st.markdown('<p class="description">Upload a cell image to check if it\'s infected with malaria parasites.</p>', unsafe_allow_html=True)

    # Sidebar Information
    st.sidebar.header("About")
    st.sidebar.info("""
        **Malaria Cell Prediction Web App** uses a deep learning model to detect malaria parasites in cell images.
        
        This project leverages Convolutional Neural Networks (CNN) to develop a web-based system for diagnosing malaria through red blood cell image analysis, offering a cost-effective, portable, and accurate alternative to traditional diagnostic tools like microscopes and Rapid Diagnostic Tests (RDT). Designed to enhance healthcare accessibility in resource-limited and remote areas, it supports the Global Technical Strategy for Malaria (GTS) 2016–2030 and Indonesia's National Action Plan for Malaria Elimination, aiming to accelerate early detection, improve diagnosis efficiency, and contribute to the global fight against malaria.
        
        **Developed by:** Achmad Ardani Prasha & Clavino Ourizqi Rachmadi
        
        Introduction To Artificial Intelligence - Mercu Buana University
    """)

    # Load the pre-trained model with caching to improve performance
    @st.cache_resource
    def load_model():
        try:
            model = tf.keras.models.load_model("my_model.h5")
            return model
        except Exception as e:
            st.sidebar.error("Error loading model: " + str(e))
            return None

    model = load_model()

    if model is not None:
        # File uploader for images
        uploaded_file = st.file_uploader("📤 Choose an image", type=["jpg", "jpeg", "png"])

        # Prediction if file is uploaded
        if uploaded_file is not None:
            try:
                # Open and display the image
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True, clamp=True)

                # Preprocess the image for model prediction
                processed_image = preprocess_image(image)

                # Make prediction with a spinner
                with st.spinner('🔍 Analyzing...'):
                    prediction = make_prediction(model, processed_image)

                # Display the prediction result
                if prediction == 0:
                    st.markdown('<p class="prediction-success">✅ **IT IS NOT INFECTED WITH MALARIA PARASITE**</p>', unsafe_allow_html=True)
                    st.write("Great news! The cell appears healthy and not infected.")
                else:
                    st.markdown('<p class="prediction-warning">⚠️ **IT IS INFECTED WITH MALARIA PARASITE**</p>', unsafe_allow_html=True)
                    st.write("Please consult a healthcare professional for further evaluation and treatment.")

            except Exception as e:
                st.error("Error processing image: " + str(e))

    # Footer with developer credits
    st.markdown("""
        <div class="footer">
            Developed by Achmad Ardani Prasha & Clavino Ourizqi Rachmadi
        </div>
        """, unsafe_allow_html=True)

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
