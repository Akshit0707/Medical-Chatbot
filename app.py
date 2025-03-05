import streamlit as st
import numpy as np
import joblib
import requests

# Load the trained cancer prediction model
model = joblib.load("cancer_prediction_model.joblib")  # Ensure the file exists

# Mistral AI API Key
MISTRAL_API_KEY = "vzJHTFVtffxg0bwH04qgXsgM5hgG6iLR"
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"

# List of 30 symptoms used in cancer prediction
symptom_names = {
    "mean_radius": 14.2,
    "mean_texture": 20.3,
    "mean_perimeter": 89.8,
    "mean_area": 462.79019607843145,
    "mean_smoothness":0.09247764705882354 ,
    "mean_compactness": 0.08008529411764707,
    "mean_concavity": 0.04605882352941176,
    "mean_concave_points": 0.02571764705882353,
    "mean_symmetry": 0.1740078431372549,
    "mean_fractal_dimension": 0.06186764705882353,
    "radius_error": 0.28408235294117645,
    "texture_error": 1.2203801120448172,
    "perimeter_error": 2.0003212885154085,
    "area_error": 21.135148459383736,
    "smoothness_error": 0.007195901960784316,
    "compactness_error": 0.02533627450980392,
    "concavity_error": 0.025996735574229688,
    "concave_points_error": 0.009857843137254902,
    "symmetry_error": 0.020584313725490196,
    "fractal_dimension_error": 0.0036360512605041998,
    "worst_radius": 13.379801120448189,
    "worst_texture": 25.691176470588236,
    "worst_perimeter": 88.34411764705883,
    "worst_area": 558.8994397759104,
    "worst_smoothness": 0.1258343137254902,
    "worst_compactness": 0.1876127450980392,
    "worst_concavity": 0.1662377226890756,
    "worst_concave_points": 0.07444411764705882,
    "worst_symmetry": 0.2702460784313725,
    "worst_fractal_dimension": 0.0794421568627451
}

# Function to interact with Mistral AI
def ask_mistral(question, context="general"):
    """Sends user input to Mistral API and retrieves a response."""
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    # Define role-based prompts
    if context == "cancer":
        system_prompt = "You are an expert oncologist. Only provide responses related to cancer prediction and medical advice. Avoid discussing any unrelated health topics."
    else:
        system_prompt = "You are a helpful AI medical assistant providing general health advice."

    data = {
        "model": "mistral-tiny",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    }

    response = requests.post(MISTRAL_URL, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return "Error fetching response from Mistral AI."

# Custom CSS for aesthetics
st.markdown("""
    <style>
        /* Ensure all text is black */
        html, body, [class*="st-"] {
            color: white !important;
            font-family: 'Arial', sans-serif;
        }

        /* Improve main container look */
        .stApp {
            background-color: light grey; /* Light grey background for a soft look */
            padding: 20px;
        }

        /* Center content with a subtle white card */
        .main-container {
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            max-width: 800px;
            margin: auto;
        }

        /* Improve headings */
        h1, h2, h3 {
            font-weight: bold;
            text-align: center;
        }

        /* Style buttons */
        .stButton > button {
            background-color: #007BFF !important;
            color: white !important;
            font-size: 16px;
            padding: 12px;
            border-radius: 8px;
            border: none;
            width: 100%;
            cursor: pointer;
        }

        .stButton > button:hover {
            background-color: #0056b3 !important;
        }

        /* Improve input fields */
        .stTextInput > div, .stNumberInput > div {
            border-radius: 8px !important;
            font-size: 16px !important;
        }

        /* Make radio buttons and checkboxes more modern */
        .stRadio > label, .stCheckbox > label {
            font-size: 16px;
            font-weight: 500;
        }
    </style>
""", unsafe_allow_html=True)

   
# Streamlit UI
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<h1 class="title">🩺 AI-Powered Medical Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="subheader">Ask me anything about health or predict your cancer risk!</h3>', unsafe_allow_html=True)

# User input for general health questions
user_input = st.text_input("💬 Ask me a health-related question:")

if user_input:
    response = ask_mistral(user_input)
    st.write(f"🤖 **AI:** {response}")


if user_input:
    response = ask_mistral(user_input)
    st.write(f"🤖 **AI:** {response}")

st.subheader("🔬 Cancer Prediction")

st.write("Please provide the values for the following medical symptoms:")

# Collect user inputs for 30 symptoms
symptoms = []
for symptom in symptom_names:
    value = st.number_input(f"{symptom.replace('_', ' ').capitalize()}:", min_value=0.0, step=0.1)
    symptoms.append(value)

# Convert input into a numpy array
input_data = np.array(symptoms).reshape(1, -1)

# Predict button
if st.button("🔍 Predict Cancer Risk"):
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        # Use LLM to explain the result
        if prediction == 1:
            result_text = f"⚠️ High Cancer Risk Detected! (Probability: {probability:.2f})"
            advice = ask_mistral("What should I do if I have a high risk of cancer?")
            st.error(result_text)
        else:
            result_text = f"✅ No Cancer Detected! (Probability: {probability:.2f})"
            advice = ask_mistral("How can I maintain good health to prevent cancer?")
            st.success(result_text)

        # Display AI-generated medical advice
        st.write(f"🤖 AI Advice: {advice}")

    except Exception as e:
        st.error(f"⚠️ Error in prediction: {e}")
        
st.markdown("</div>", unsafe_allow_html=True)
