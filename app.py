import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

# Load the trained model
pipe_lr = joblib.load(open("model/text_emotion.pkl", "rb"))

# Emoji dictionary for emotions
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", 
    "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"
}

# Function to predict emotion
def predict_emotions(docx):
    return pipe_lr.predict([docx])[0]

# Function to get prediction probabilities
def get_prediction_proba(docx):
    return pipe_lr.predict_proba([docx])

# Main function for the Streamlit app
def main():
    st.set_page_config(page_title="Text Emotion Detector", page_icon="🔍", layout="centered")
    
    st.markdown("""
        <style>
            .main-container {
                background-color: #f4f4f4;
                padding: 20px;
                border-radius: 10px;
            }
            .title {
                font-size: 32px;
                font-weight: bold;
                text-align: center;
                color: #4A90E2;
            }
            .subheader {
                text-align: center;
                font-size: 20px;
                color: #333;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='title'>Text Emotion Detection</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader'>Detect emotions in text with AI-powered insights</div>", unsafe_allow_html=True)
    st.write("---")
    
    with st.form(key='emotion_form'):
        raw_text = st.text_area("Enter Your Text Here", height=150, placeholder="Type something...")
        submit_text = st.form_submit_button(label='Analyze Emotion')
    
    if submit_text:
        if raw_text.strip():
            col1, col2 = st.columns(2)
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            with col1:
                st.success("Original Text")
                st.write(f"✏️ *{raw_text}*")
                
                st.success("Prediction")
                emoji_icon = emotions_emoji_dict.get(prediction, "❓")
                st.markdown(f"<h3 style='text-align: center;'>{prediction} {emoji_icon}</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; font-size: 18px;'>Confidence: {np.max(probability):.2f}</p>", unsafe_allow_html=True)

            with col2:
                st.success("Prediction Probability")
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["Emotion", "Probability"]
                
                chart = alt.Chart(proba_df_clean).mark_bar(cornerRadius=6).encode(
                    x=alt.X('Emotion', sort='-y', title=""),
                    y=alt.Y('Probability', title="Confidence Level"),
                    color=alt.Color('Emotion', scale=alt.Scale(scheme='category10'))
                ).properties(
                    width=300, height=250
                )
                
                st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("⚠️ Please enter some text before submitting.")

if __name__ == '__main__':
    main()