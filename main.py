import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

def main():
    # Apply enhanced custom CSS for a minimalist look
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
            
            body {
                background-color: #f4f7f9;
                font-family: 'Inter', sans-serif;
                color: #333333;
            }
            .title {
                color: #1A73E8;
                text-align: center;
                font-size: 42px;
                margin-bottom: 5px;
            }
            .description {
                color: #4A5568;
                text-align: center;
                font-size: 18px;
                margin-bottom: 40px;
            }
            .footer {
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background-color: #ffffff;
                color: #6c757d;
                text-align: center;
                padding: 8px 0;
                font-size: 14px;
                box-shadow: 0 -2px 8px rgba(0, 0, 0, 0.1);
            }
            .prediction-success {
                color: #2ECC71;
                font-size: 20px;
                text-align: center;
                margin-top: 20px;
            }
            .prediction-warning {
                color: #E74C3C;
                font-size: 20px;
                text-align: center;
                margin-top: 20px;
            }
            .upload-section {
                text-align: center;
                margin-top: 20px;
                margin-bottom: 40px;
            }
            .upload-button > label {
                background-color: #1A73E8;
                color: white;
                padding: 12px 24px;
                border-radius: 4px;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            .upload-button > label:hover {
                background-color: #145DB2;
            }
            .stButton button {
                background-color: #1A73E8;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
            }
            .stButton button:hover {
                background-color: #145DB2;
            }
        </style>
        """, unsafe_allow_html=True)

    # Title and Description
    st.markdown('<h1 class="title">🦠 Malaria Cell Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="description">Upload a cell image to check if it\'s infected with malaria parasites.</p>', unsafe_allow_html=True)

    # Sidebar Information
    st.sidebar.header("About")
    st.sidebar.info("""
        **Malaria Cell Prediction Web App** uses a deep learning model to detect malaria parasites in cell images.
        
        This project leverages CNN for diagnosing malaria through image analysis, contributing to global healthcare improvements.
        
                    
        **Developed by:** Achmad Ardani Prasha & Clavino Ourizqi Rachmadi

        *Introduction To Artificial Intelligence - Mercu Buana University*
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
        # Centralized uploader section with margin
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload your image below:", type=["jpg", "jpeg", "png"], key="uploader")
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            try:
                # Display the image
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)

                # Preprocess the image for model prediction
                processed_image = preprocess_image(image)

                # Make prediction with a spinner
                with st.spinner('🔍 Analyzing...'):
                    prediction = make_prediction(model, processed_image)

                # Display the prediction result
                if prediction == 0:
                    st.markdown('<p class="prediction-success">✅ <strong>NOT INFECTED WITH MALARIA PARASITE</strong></p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="prediction-warning">⚠️ <strong>INFECTED WITH MALARIA PARASITE</strong></p>', unsafe_allow_html=True)

            except Exception as e:
                st.error("Error processing image: " + str(e))

    # Footer with developer credits
    st.markdown("""
        <div class="footer">
            Developed by Achmad Ardani Prasha & Clavino Ourizqi Rachmadi | Mercu Buana University
        </div>
        """, unsafe_allow_html=True)

def preprocess_image(image):
    """
    Preprocesses the image to be suitable for model prediction.
    Resize to the input shape expected by the model and normalize pixel values.
    """
    resized_image = image.resize((64, 64))
    img_array = np.array(resized_image)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    normalized_image = img_array / 255.0
    processed_image = np.expand_dims(normalized_image, axis=0)
    return processed_image

def make_prediction(model, image):
    """
    Takes the model and preprocessed image as input, returns the predicted class.
    """
    predictions = model.predict(image)
    if predictions.shape[-1] == 1:
        predicted_class = int(predictions[0][0] > 0.5)
    else:
        predicted_class = np.argmax(predictions, axis=1)[0]
    return predicted_class

if __name__ == "__main__":
    main()
    