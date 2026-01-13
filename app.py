import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Jordan Antiquities",
    page_icon="🇯🇴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR INFO ---
with st.sidebar:
    # You can replace this URL with a logo of your choice if you have one
    st.image("https://cdn-icons-png.flaticon.com/512/3208/3208726.png", width=100)
    st.title("About")
    st.info(
        """
        Jordan Antiquities is an AI-powered system designed to 
        identify archaeological sites in Jordan using Deep Learning (ResNet152).
        
        **Supported Sites:**
        - Petra
        - Jerash
        - Wadi Rum
        - Roman Amphitheater
        - Umm Qais
        - Ajloun Castle
        """
    )
    st.write("---")

# --- MODEL CONFIGURATION ---
class_names = [
    "Ajloun", 
    "Jerash", 
    "Petra", 
    "UmmQais",            
    "RomanAmphitheater",  
    "WadiRum"
]
IMG_SIZE = (224, 224) 

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model.h5')
    return model

model = load_model()

# --- MAIN PAGE UI ---
# UPDATED: Removed icon and added space
st.title("Jordan Antiquities")
st.markdown("### Upload a photo to identify the site")
st.write("---")

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Your Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True, caption="Uploaded Photo")

    # PREDICTION LOGIC
    image_processed = ImageOps.fit(image, IMG_SIZE, Image.Resampling.LANCZOS)
    image_processed = image_processed.convert("RGB")
    img_array = np.array(image_processed)
    img_array = img_array.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    score = prediction[0] 
    
    top_class_index = np.argmax(score)
    top_class = class_names[top_class_index]
    confidence = 100 * np.max(score)

    # DISPLAY RESULTS IN COLUMN 2
    with col2:
        st.subheader("Analysis Result")
        st.write("") 
        
        st.metric(label="Identified Location", value=top_class)
        
        st.write(f"Confidence: **{confidence:.1f}%**")
        st.progress(int(confidence))
        
        if confidence > 80:
            st.success(f"✅ We are confident this is {top_class}.")
        elif confidence > 50:
            st.warning("⚠️ Moderate confidence. Ensure the image is clear.")
        else:
            st.error("❓ Low confidence. This might not be a supported site.")

        with st.expander(f"Learn more about {top_class}"):
            st.write(f"Click below to see {top_class} on Google Maps.")
            st.markdown(f"[📍 Open in Google Maps](https://www.google.com/maps/search/?api=1&query={top_class}+Jordan)")