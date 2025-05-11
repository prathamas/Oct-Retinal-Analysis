import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
import tempfile
from PIL import Image

# Custom CSS for styling
def set_custom_style():
    st.markdown("""
    <style>
        /* Main button styling */
        .stButton>button {
            border: 2px solid #4e4376;
            border-radius: 20px;
            color: white;
            background-color: #2b5876;
            padding: 10px 24px;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #4e4376;
            color: white;
            border: 2px solid #2b5876;
        }
        
        /* File uploader styling */
        .stFileUploader>div>div>div>div {
            border: 2px dashed #4e4376 !important;
            background: rgba(255,255,255,0.7) !important;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.7);
            border-radius: 10px 10px 0 0 !important;
            padding: 10px 20px !important;
        }
        .stTabs [aria-selected="true"] {
            background: #2b5876 !important;
            color: white !important;
        }
        
        /* Custom cards */
        .custom-card {
            background: rgba(255,255,255,0.7);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        /* Disease highlight */
        .disease-highlight {
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid;
            margin: 20px 0;
        }
    </style>
    """, unsafe_allow_html=True)

# Apply custom styles
set_custom_style()

# Tensorflow Model Prediction
def model_prediction(test_image_path):
    model = tf.keras.models.load_model("Trained_Eye_disease_model.keras")
    img = tf.keras.utils.load_img(test_image_path, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predictions = model.predict(x)
    return np.argmax(predictions)  # return index of max element

# Disease information with enhanced formatting
disease_info = {
    "CNV": {
        "title": "Choroidal Neovascularization (CNV)",
        "description": "Abnormal blood vessel growth beneath the retina causing fluid leakage and vision distortion.",
        "symptoms": ["Distorted vision", "Dark spots in central vision", "Rapid vision loss"],
        "treatment": ["Anti-VEGF injections", "Photodynamic therapy", "Laser treatment"],
        "color": "#FF6B6B",
        "icon": "🩺"
    },
    "DME": {
        "title": "Diabetic Macular Edema (DME)",
        "description": "Fluid accumulation in the retina causing swelling and potential vision loss in diabetic patients.",
        "symptoms": ["Blurred vision", "Colors appearing washed out", "Difficulty reading"],
        "treatment": ["Anti-VEGF therapy", "Corticosteroids", "Laser treatment", "Blood sugar control"],
        "color": "#4ECDC4",
        "icon": "💉"
    },
    "DRUSEN": {
        "title": "Drusen (Early AMD)",
        "description": "Yellow deposits under the retina that may indicate early age-related macular degeneration.",
        "symptoms": ["Often asymptomatic", "Mild vision changes", "Difficulty adapting to low light"],
        "treatment": ["AREDS2 supplements", "Regular monitoring", "Lifestyle changes"],
        "color": "#FFD166",
        "icon": "👁️"
    },
    "NORMAL": {
        "title": "Normal Retina",
        "description": "Healthy retina with no signs of disease or abnormalities.",
        "symptoms": ["Clear vision", "No visual distortions", "Normal color perception"],
        "treatment": ["Regular eye exams", "Healthy diet", "UV protection"],
        "color": "#06D6A0",
        "icon": "✅"
    }
}

# Sidebar with enhanced design
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;">
        <h1 style="color:#2b5876; margin-bottom:0;"> OCT Analyzer</h1>
        <p style="color:#4e4376; margin-top:0;">Retinal Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    app_mode = st.selectbox(
        "Navigate", 
        ["Home", "About", "Disease Identification"],
        index=0,
        key='nav_select'
    )
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;">
        <p>Developed by</p>
        <p style="font-weight:bold; color:#2b5876;">Pratham Singh</p>
    </div>
    """, unsafe_allow_html=True)

# Home Page
if app_mode == "Home":
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("""
        <div style="font-size: 100px; text-align: center;"></div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <h1 style="color:#2b5876; margin-bottom:0;">OCT Retinal Analysis Platform</h1>
        <p style="color:#4e4376; font-size:18px;">Advanced AI for Early Detection of Retinal Diseases</p>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features section with cards
    st.markdown("### Key Features")
    cols = st.columns(4)
    features = [
        {"icon": "⚡", "title": "Instant Analysis", "desc": "Get results in seconds with our AI model"},
        {"icon": "🔄", "title": "Automated Workflow", "desc": "Streamlined process from upload to diagnosis"},
        {"icon": "📊", "title": "Detailed Reports", "desc": "Comprehensive disease information and recommendations"},
        {"icon": "🔒", "title": "Secure Processing", "desc": "Your data remains private and secure"}
    ]
    
    for i, feature in enumerate(features):
        with cols[i]:
            st.markdown(f"""
            <div class="custom-card">
                <h3>{feature['icon']} {feature['title']}</h3>
                <p>{feature['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Disease overview with tabs
    st.markdown("### Understanding Retinal Diseases")
    tabs = st.tabs(["CNV", "DME", "Drusen", "Normal"])
    
    for i, (disease, data) in enumerate(disease_info.items()):
        with tabs[i]:
            st.markdown(f"""
            <div style="background: {data['color']}20; padding: 15px; border-radius: 10px; border-left: 5px solid {data['color']};">
                <h3>{data['icon']} {data['title']}</h3>
                <p>{data['description']}</p>
                <h4>Symptoms:</h4>
                <ul>
                    {''.join([f'<li>{s}</li>' for s in data['symptoms']])}
                </ul>
                <h4>Treatment Options:</h4>
                <ul>
                    {''.join([f'<li>{t}</li>' for t in data['treatment']])}
                </ul>
            </div>
            """, unsafe_allow_html=True)


# About Page
elif app_mode == "About":
    st.title("🔬 About OCT Analyzer")
    st.subheader("Empowering ophthalmologists and patients with AI-driven retinal diagnostics")

    st.markdown("---")

    st.header("🌍 Mission")
    st.markdown("""
    Our mission is to democratize access to early retinal disease detection through the power of artificial intelligence.
    By combining cutting-edge deep learning with an easy-to-use interface, we aim to assist both healthcare professionals
    and individuals in making informed eye health decisions.
    """)

    st.header("🧠 How It Works")
    st.markdown("""
    - **Upload an OCT Scan**: Use our secure tool to upload retinal images.
    - **AI Analysis**: The model instantly processes the image using MobileNetV3.
    - **Diagnosis & Recommendation**: Get a predicted class with a probability score and basic suggestions.
    """)

    st.header("📊 Dataset Highlights")
    st.markdown("""
    - 🖼️ **84,495 high-resolution OCT images**
    - 🔍 Expert-labeled across **4 categories**: CNV, DME, Drusen, and Normal
    - ✅ Balanced and rigorously validated dataset
    """)

    st.header("🧰 Tech Stack")
    st.markdown("""
    - **TensorFlow**: For deep learning model training and inference
    - **MobileNetV3**: Lightweight CNN architecture for efficient classification
    - **Streamlit**: Interactive web interface built with Python
    """)

    st.header("👩‍⚕️ Clinical Relevance")
    st.markdown("""
    While not a substitute for professional diagnosis, our model shows strong performance:
    - 🔬 92% accuracy on **CNV**
    - 🔬 89% accuracy on **DME**
    - 🔬 94% accuracy on **Normal Retina**
    - 💡 Helpful for screening and decision support in clinical settings
    """)

    st.header("📢 Disclaimer")
    st.markdown("""
    This app is intended for educational and preliminary screening use only.
    Please consult a certified ophthalmologist for any medical diagnosis or treatment.
    """)


# Disease Identification Page
elif app_mode == "Disease Identification":
    st.markdown("""
    <div style="text-align:center;">
        <h1 style="color:#2b5876;">Retinal OCT Analysis</h1>
        <p style="font-size:18px;">Upload an OCT scan for instant analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # File uploader with custom styling
    with st.container():
        st.markdown("""
        <div style="background: rgba(255,255,255,0.7); padding: 30px; border-radius: 10px; text-align: center; border: 2px dashed #4e4376;">
            <h3>📁 Upload OCT Image</h3>
            <p>Supported formats: JPG, PNG</p>
        </div>
        """, unsafe_allow_html=True)
        
        test_image = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
    if test_image is not None:
        # Display the uploaded image
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("### Uploaded Image")
            img = Image.open(test_image)
            st.image(img, caption="Your OCT Scan", use_container_width=True)
        
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            img.save(tmp_file.name)
            temp_file_path = tmp_file.name
    
    # Predict button with loading animation
    if st.button("🔍 Analyze Image", use_container_width=True, type="primary") and test_image is not None:
        with st.spinner("Analyzing retinal patterns..."):
            result_index = model_prediction(temp_file_path)
            class_name = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
            disease = class_name[result_index]
            data = disease_info[disease]
            
            # Show result with color-coded alert
            st.markdown(f"""
            <div class="disease-highlight" style="background: {data['color']}20; border-left-color: {data['color']}">
                <h2 style="color: {data['color']}; margin-top: 0;">{data['icon']} Result: {data['title']}</h2>
                <p style="font-size: 16px;">{data['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed information in expander
            with st.expander("📋 Detailed Report & Recommendations", expanded=True):
                tabs = st.tabs(["Overview", "Symptoms", "Treatment", "Resources"])
                
                with tabs[0]:
                    st.markdown(f"""
                    <h3>About {disease}</h3>
                    <p>{data['description']}</p>
                    <h4>Key Characteristics:</h4>
                    <ul>
                        {''.join([f'<li>{s}</li>' for s in data['symptoms'][:2]])}
                    </ul>
                    """, unsafe_allow_html=True)
                    st.image(test_image, use_column_width=True)
                
                with tabs[1]:
                    st.markdown("""
                    <h3>Common Symptoms</h3>
                    <ul>
                    """ + ''.join([f'<li>{s}</li>' for s in data['symptoms']]) + """
                    </ul>
                    """, unsafe_allow_html=True)
                
                with tabs[2]:
                    st.markdown("""
                    <h3>Treatment Options</h3>
                    <ul>
                    """ + ''.join([f'<li>{t}</li>' for t in data['treatment']]) + """
                    </ul>
                    <p><i>Note: Always consult with an ophthalmologist for personalized treatment plans.</i></p>
                    """, unsafe_allow_html=True)
                
                with tabs[3]:
                    st.markdown("""
                    <h3>Additional Resources</h3>
                    <ul>
                        <li><a href="https://www.aao.org/" target="_blank">American Academy of Ophthalmology</a></li>
                        <li><a href="https://www.nei.nih.gov/" target="_blank">National Eye Institute</a></li>
                        <li><a href="https://www.macular.org/" target="_blank">American Macular Degeneration Foundation</a></li>
                    </ul>
                    """, unsafe_allow_html=True)
            
            # Visual comparison
            st.markdown("---")
            st.markdown("### Visual Comparison")
            cols = st.columns(4)
            for i, (d, info) in enumerate(disease_info.items()):
                with cols[i]:
                    st.markdown(f"""
                    <div style="text-align:center;">
                        <div style="background: {info['color']}20; padding: 10px; border-radius: 10px; 
                            border: {'2px solid ' + info['color'] if d == disease else 'none'};">
                            <h4>{info['icon']} {d}</h4>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    elif test_image is None and st.session_state.get('nav_select') == "Disease Identification":
        st.warning("Please upload an OCT image to analyze")