import streamlit as st
import numpy as np
import joblib
import requests

# Load the trained cancer prediction model
model = joblib.load("cancer_prediction_model.joblib")  # Ensure the file exists

# Mistral AI API Key (Replace with actual key)
MISTRAL_API_KEY = "vzJHTFVtffxg0bwH04qgXsgM5hgG6iLR"
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"

# Feature questions (with numbering)
feature_questions = {
    "mean_radius": "1. Have you noticed any unusual lumps or growths in your body?",
    "mean_texture": "2. Have you experienced any skin texture changes in affected areas?",
    "mean_perimeter": "3. Have you observed an increase in the size of any lumps over time?",
    "mean_area": "4. Have you had any swelling in a particular body area?",
    "mean_smoothness": "5. Does your skin in affected areas feel rough or bumpy?",
    "mean_compactness": "6. Have you noticed hard or firm masses under your skin?",
    "mean_concavity": "7. Do any lumps on your body have an inward curve or dimple?",
    "mean_concave_points": "8. Are there areas on your skin that appear sunken or hollow?",
    "mean_symmetry": "9. Have you observed an asymmetrical shape in any lumps?",
    "mean_fractal_dimension": "10. Have you noticed complex patterns or irregular edges in skin changes?",
    "radius_error": "11. Has the size of any lumps changed unpredictably?",
    "texture_error": "12. Have you had skin changes that feel different than normal?",
    "perimeter_error": "13. Has the border of a skin lesion changed over time?",
    "area_error": "14. Have you experienced spreading of affected areas?",
    "smoothness_error": "15. Does your skin feel rougher in certain regions?",
    "compactness_error": "16. Have you noticed tightness or hardness in affected skin?",
    "concavity_error": "17. Are any affected areas forming depressions or indentations?",
    "concave_points_error": "18. Are there multiple sunken spots in the same area?",
    "symmetry_error": "19. Have you noticed an imbalance in affected areas of your body?",
    "fractal_dimension_error": "20. Do any affected skin areas appear to have jagged or uneven edges?",
    "worst_radius": "21. Has the largest affected area increased in size recently?",
    "worst_texture": "22. Has your skin developed a rough or scaly texture?",
    "worst_perimeter": "23. Have you noticed a significant change in the borders of any lumps?",
    "worst_area": "24. Has an affected area expanded rapidly?",
    "worst_smoothness": "25. Has your skin become increasingly rough?",
    "worst_compactness": "26. Has the density or hardness of any lumps increased?",
    "worst_concavity": "27. Are there deep indentations in any affected areas?",
    "worst_concave_points": "28. Are there several deep hollow points in a lump?",
    "worst_symmetry": "29. Have you noticed a significant asymmetry in any affected areas?",
    "worst_fractal_dimension": "30. Do affected skin areas have highly irregular edges?"
}

# Initialize session state
if "question_index" not in st.session_state:
    st.session_state.question_index = 0
if "responses" not in st.session_state:
    st.session_state.responses = {}

# Convert questions dictionary to a list
feature_keys = list(feature_questions.keys())

# Ensure index is within range
if st.session_state.question_index >= len(feature_keys):
    st.session_state.question_index = len(feature_keys) - 1  

# Get current question
current_feature = feature_keys[st.session_state.question_index]
current_question = feature_questions[current_feature]

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>🩺 AI-Powered Medical Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Ask me anything about health or predict your cancer risk!</h3>", unsafe_allow_html=True)

# AI Chatbot: User can ask any health-related question
user_input = st.text_input("💬 Ask me a health-related question:")

# Function to interact with Mistral AI
def ask_mistral(question):
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "mistral-tiny",
        "messages": [
            {"role": "system", "content": "You are an AI medical assistant providing health advice."},
            {"role": "user", "content": question}
        ]
    }
    response = requests.post(MISTRAL_URL, headers=headers, json=data)
    return response.json()["choices"][0]["message"]["content"] if response.status_code == 200 else "Error fetching AI response."

if user_input:
    response = ask_mistral(user_input)
    st.write(f"🤖 **AI:** {response}")
    
st.markdown("<hr>", unsafe_allow_html=True)  # Divider line

# **Cancer Prediction Section**
st.markdown("<h2 style='text-align: center;'>🔬 AI Cancer Prediction Chatbot</h2>", unsafe_allow_html=True)
st.write("Hi! I'm an AI chatbot here to assess your cancer risk. Let's begin!")

# Show progress
st.write(f"**Question {st.session_state.question_index + 1} of {len(feature_questions)}**")

# Display the current question
response = st.radio(current_question, ("No", "Yes"), key=current_feature)

# "Next" button
if st.button("Next"):
    st.session_state.responses[current_feature] = 1 if response == "Yes" else 0  # Store response
    if st.session_state.question_index < len(feature_keys) - 1:
        st.session_state.question_index += 1  # Move to next question
    st.experimental_rerun()  # Refresh UI to show next question

# Predict Cancer Risk Button (only when all questions are answered)
if len(st.session_state.responses) == len(feature_questions):
    if st.button("Predict Cancer Risk"):
        responses = list(st.session_state.responses.values())

        # Define high-risk features
        high_risk_features = ["mean_radius", "worst_radius", "mean_perimeter", "worst_perimeter", "mean_area", "worst_area"]

        # Calculate risk scores
        risk_score = sum(responses)  # Total Yes responses
        high_risk_score = sum(
            1 for i, feature in enumerate(feature_questions.keys()) if feature in high_risk_features and responses[i] == 1
        )

        # Determine cancer risk
        if high_risk_score >= 3 or risk_score >= 10:
            result = "⚠️ High Cancer Risk - Consult a Doctor"
        elif risk_score >= 5:
            result = "⚠️ Moderate Cancer Risk - Further Tests Recommended"
        else:
            result = "✅ Low Cancer Risk - No Immediate Concern"

        # Display Result
        st.subheader("📊 Prediction Result:")
        st.write(result)
