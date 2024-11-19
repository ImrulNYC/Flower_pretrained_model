import streamlit as st
import os
from werkzeug.utils import secure_filename
from prediction import predict_flower
from datetime import datetime
import pytz

# Set page configuration
st.set_page_config(
    page_title="Flower Identification App",
    page_icon="🌼",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Function to get current Eastern Time
def get_current_eastern_time():
    eastern = pytz.timezone('US/Eastern')
    return datetime.now(eastern).strftime("%Y-%m-%d %H:%M:%S")

# Add button for Night Mode and Light Mode
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

if st.button("Toggle Night Mode" if st.session_state.theme == 'light' else "Toggle Light Mode"):
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'

# Custom CSS for styling based on theme
if st.session_state.theme == 'light':
    background_gradient = 'linear-gradient(to bottom right, #e0f7fa, #f0f8ff)'
    text_color = '#4CAF50'
    background_color = '#ffffffcc'
else:
    background_gradient = 'linear-gradient(to bottom right, #000000, #434343)'
    text_color = '#ffffff'
    background_color = '#000000cc'

st.markdown(
    f"""
    <style>
    .main {{
        background: {background_gradient};
        padding: 20px;
        border-radius: 15px;
    }}
    .stButton>button {{
        background-color: {text_color};
        color: white;
        border-radius: 8px;
        font-size: 16px;
        padding: 10px 24px;
    }}
    .stTextInput>div>input {{
        border-radius: 8px;
    }}
    img.uploaded-image {{
        border: 5px solid {text_color};
        border-radius: 15px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app title 
st.markdown(
    f"""
    <div style="text-align: center; background: {background_color}; padding: 20px; border-radius: 15px;">
        <h1 style="color: {text_color}; font-size: 3em;">Flower Identification App 🌼</h1>
        <p style="font-size: 1.2em; color: #555;">Upload a flower image to discover its name </p>
        <p style="font-size: 1em; color: #777;">Current Eastern Time: {get_current_eastern_time()}</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar for additional pages
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Developer Info"])

if page == "Home":
    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], help="Upload an image file (JPG, JPEG, PNG) to identify the flower")

    if uploaded_file is not None:
        # Display the uploaded image using st.image() with updated parameter
        st.image(uploaded_file, caption='Uploaded Image', use_container_width=True, output_format='auto')

        # Save the uploaded file for prediction purposes
        UPLOAD_FOLDER = 'static/uploads'
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)

        filename = secure_filename(uploaded_file.name)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Predict the flower type
        with st.spinner('Identifying the flower...'):
            predicted_label, confidence = predict_flower(file_path)
        
        if predicted_label:
            st.success(f"Predicted Flower: {predicted_label} with {confidence:.2f}% confidence.")
        else:
            st.warning("The flower cannot be confidently recognized. Please try another image.")

    else:
        st.markdown(
            f"""
            <div style="text-align: center; margin-top: 50px; background: {background_color}; padding: 20px; border-radius: 15px;">
                <p style="font-size: 1.2em; color: #777;">Upload an image to get started and explore the beauty of flowers!</p>
            </div>
            """,
            unsafe_allow_html=True
        )

elif page == "Developer Info":
    # Developer information page
    st.markdown(
        f"""
        <div style="text-align: center; background: {background_color}; padding: 20px; border-radius: 15px;">
            <h2 style="color: {text_color};">Developer Team</h2>
            <p style="font-size: 1.2em; color: #555;">This app was developed by a CTP Team:</p>
            <ul style="list-style-type: none; font-size: 1.2em; color: #777;">
                <li>1. Jessica</li>
                <li>2. Mansur</li>
                <li>3.  Zahava </li>
                <li>4. Imrul</li>
            </ul>
            <div style="margin-top: 20px;">
    <p style="font-size: 1em; color: #999;">
        Model: This is a pretrained ViT-16 model for flower classification, with additional fine-tuning. The website was designed by us. Check out our other model, which was built from scratch: <a href="https://www.google.com" style="color: #999; text-decoration: underline;">here</a>

</div>

        
        """,
        unsafe_allow_html=True
    )
