# 🇯🇴 Jordan Antiquities AI

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](PASTE_YOUR_HUGGING_FACE_LINK_HERE)
![Python](https://img.shields.io/badge/Python-3.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange)
![License](https://img.shields.io/badge/License-MIT-green)

An AI-powered Computer Vision system designed to preserve and identify Jordan's rich archaeological heritage. This application uses **Deep Learning (ResNet152)** to classify images of 6 major historical sites in Jordan with high accuracy.

## 🔴 Live Demo
**👉 [Click here to try the App](https://huggingface.co/spaces/JawadHaddad/JordanAntiquities)**

---

## 📸 Screenshots
<img width="1620" height="942" alt="Screenshot 2026-01-13 065643" src="https://github.com/user-attachments/assets/6aef5ea2-8c04-476f-acf4-1ef01e1b5d19" />

---

## 🏛️ Supported Sites
The model has been trained on a custom dataset to identify:
* **Petra** (The Rose City)
* **Jerash** (Roman Ruins)
* **Wadi Rum** (The Valley of the Moon)
* **Amman Roman Theater**
* **Umm Qais** (Gadara)
* **Ajloun Castle**

## 🧠 Technical Details
* **Model Architecture:** ResNet152 (Transfer Learning from ImageNet)
* **Training:** Fine-tuned on a custom dataset of ~1,000 images.
* **Deployment:** Streamlit frontend, hosted on Hugging Face Spaces.
* **Performance:** Achieved ~99% accuracy on the test set.

## 🚀 How to Run Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_GITHUB_USERNAME/Jordan-Heritage-Classifier.git](https://github.com/YOUR_GITHUB_USERNAME/Jordan-Heritage-Classifier.git)
