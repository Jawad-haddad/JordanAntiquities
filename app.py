import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- CONFIGURATION ---
# IMPORTANT: Update this list to match the alphabetical order of your training folders
class_names = [
    "Ajloun", 
    "Jerash", 
    "Petra", 
    "RomanAmphitheater", 
    "UmmQais", 
    "WadiRum"
]

IMG_SIZE = (224, 224)
st.set_page_config(page_title="Jordan Heritage AI", page_icon="🇯🇴")

@st.cache_resource
def load_model():
    # This loads the model once and keeps it in memory
    model = tf.keras.models.load_model('model.h5')
    return model

# Load the model immediately
with st.spinner('Loading Model... Please wait...'):
    model = load_model()

st.title("🇯🇴 Jordan Heritage Site Classifier")
st.markdown("Upload a photo of a site (e.g., Petra, Jerash) and the AI will identify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image to fit the model
    # 1. Convert to RGB (removes alpha channel if PNG)
    image = ImageOps.fit(image, IMG_SIZE, Image.Resampling.LANCZOS)
    image = image.convert("RGB")
    
    # 2. Convert to numpy array
    img_array = np.array(image)
    
    # 3. Normalize (assuming you trained with values 0-1. If you didn't, remove the / 255.0)
    img_array = img_array.astype(np.float32) / 255.0
    
    # 4. Create a batch (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    if st.button('Identify Location'):
        prediction = model.predict(img_array)
        score = tf.nn.softmax(prediction[0])
        
        # Get the highest confidence class
        top_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)
        
        st.success(f"Prediction: **{top_class}**")
        st.info(f"Confidence: **{confidence:.2f}%**")