# 🧠 OCT Retinal Analysis Platform

A deep learning-powered web application for automated classification of retinal OCT scans into four key categories: **CNV**, **DME**, **Drusen**, and **Normal**. Built with **Streamlit** and powered by a **MobileNetV3** model.

---

## 🚀 Features

- 📸 **Image Upload**: Easily upload OCT scan images for instant classification.
- 🤖 **Disease Prediction**: Accurate model predictions trained on 84,495 expertly-labeled OCT images.
- 🩺 **Diagnosis Recommendations**: Learn about the predicted condition with rich, medically-backed recommendations.
- 🌐 **Streamlit UI**: Clean and intuitive web interface for clinicians and researchers.

---

## 🏥 Diseases Covered

- **CNV** (Choroidal Neovascularization): Subretinal fluid and neovascular membranes.
- **DME** (Diabetic Macular Edema): Retinal thickening and intraretinal fluid.
- **Drusen**: Deposits indicative of early AMD (Age-related Macular Degeneration).
- **Normal**: Clear retinal structure with preserved foveal contour.

---

## 🧠 Model

The model is based on **MobileNetV3**, fine-tuned on the **OCT2017 dataset**, which includes 84k+ labeled images across 4 classes. Images underwent multi-tier clinical validation to ensure label accuracy.

---

## 📂 Dataset Source

- **Source**: OCT2017 dataset
- **Size**: 84,495 labeled images
- **Classes**: CNV, DME, Drusen, Normal
- **Collected From**:
  - UC San Diego, USA
  - Shanghai First People’s Hospital, China
  - Beijing Tongren Eye Center, China

---
# OCT Retinal Analysis Platform

This project is a machine learning model for analyzing OCT retinal images.

## Deployed App
You can access the deployed application here: [OCT Retinal Analysis App](https://oct-retinal-analysis-qhswrqfmxagdu7zkpxrqy8.streamlit.app)

## Description
This platform helps in detecting retinal diseases from OCT scans.

## 🛠️ Setup Instructions

### 🔧 Requirements

Make sure you have Python 3.7+ installed.

### 📦 Install Dependencies

```bash
pip install -r requirements.txt

### ▶️ Run the App

streamlit run app.py

### 📁 Project Structure

oct-retinal-analysis/
│
├── app.py                  # Main Streamlit app
├── Trained_Model.keras     # Pre-trained MobileNetV3 model      
├── requirements.txt
└── README.md

### 🙏 Acknowledgements
This project was developed with guidance from online tutorials and customized to fit a medical diagnostic use case.

Based on a YouTube tutorial by ["SPOTLESS TECH"]

Enhanced using support from ChatGPT and Streamlit documentation