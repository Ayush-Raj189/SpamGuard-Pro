import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK data resources
for resource in ['punkt', 'punkt_tab', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource != 'stopwords' else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)


ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load models
try:
    tfidf = pickle.load(open('vectorizer.pkl','rb'))
    model = pickle.load(open('model.pkl','rb'))
except FileNotFoundError:
    st.error("Model files not found. Please make sure 'vectorizer.pkl' and 'model.pkl' are in the same directory.")
    st.stop()

# Responsive CSS + animation
st.markdown("""
<style>
    /* Base styles */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: #f8f9fa;
        margin: 0; padding: 0;
    }
    .main-header {
        font-size: 2.8rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 800;
        animation: pulse 3s infinite;
    }
    @keyframes pulse {
        0%, 100% {text-shadow: 0 0 10px #FF4B4B;}
        50% {text-shadow: 0 0 20px #FF2B2B;}
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border: none;
        padding: 0.85rem 1.8rem;
        border-radius: 0.6rem;
        font-size: 1.2rem;
        cursor: pointer;
        width: 100%;
        transition: all 0.35s ease;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(255,75,75,0.5);
    }
    .stButton>button:hover {
        background-color: #FF2B2B;
        transform: translateY(-4px);
        box-shadow: 0 8px 12px rgba(255,43,43,0.7);
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 0.6rem;
        margin-top: 1.8rem;
        text-align: center;
        font-weight: bold;
        font-size: 1.3rem;
        user-select: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .spam-result {
        background-color: #FFE6E6;
        color: #FF4B4B;
        border: 2px solid #FF4B4B;
    }
    .ham-result {
        background-color: #E6FFE6;
        color: #4CAF50;
        border: 2px solid #4CAF50;
    }
    .info-box {
        background-color: #E6F3FF;
        padding: 1rem 1.2rem;
        border-radius: 0.6rem;
        border-left: 5px solid #1E88E5;
        margin-bottom: 1.8rem;
        font-size: 1rem;
        box-shadow: 0 2px 4px rgba(30,136,229,0.1);
    }
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.2rem;
            margin-bottom: 1rem;
        }
        .stButton>button {
            padding: 0.7rem 1.2rem;
            font-size: 1.1rem;
        }
        .result-box {
            padding: 1.2rem;
            font-size: 1.15rem;
        }
        .stTextArea textarea {
            min-height: 140px;
        }
    }
    /* Tablet responsiveness */
    @media (max-width: 1024px) and (min-width: 769px) {
        .main-header {
            font-size: 2.5rem;
        }
    }
    html {
        font-size: 16px;
    }
    .streamlit-expanderHeader {
        font-size: 1.15rem !important;
        padding: 0.9rem !important;
    }
    /* Footer styling */
    .footer {
        margin-top: 3rem;
        padding: 1rem 0;
        border-top: 1px solid #ddd;
        font-size: 0.95rem;
        text-align: center;
        color: #555;
    }
    .footer a {
        color: #FF4B4B;
        text-decoration: none;
        font-weight: 600;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    /* Heart beat animation */
    .heart {
        color: #FF4B4B;
        animation: heartbeat 1.5s ease-in-out infinite;
        display: inline-block;
        margin-left: 0.3rem;
    }
    @keyframes heartbeat {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.3); }
    }
</style>
""", unsafe_allow_html=True)

# App layout
st.markdown('<h1 class="main-header">📱 SMS Spam Classifier</h1>', unsafe_allow_html=True)

# Information box
with st.expander("ℹ️ About this app", expanded=False):
    st.markdown("""
    This app uses machine learning to classify SMS messages as **Spam** or **Not Spam**.
    - Enter your message in the text box below
    - Click the 'Predict' button to analyze
    - The app will show whether the message is likely spam
    """)

# Input section
st.subheader("🔍 Enter your message")
input_sms = st.text_area("",
                        placeholder="Type or paste your SMS message here...", 
                        height=150,
                        label_visibility="collapsed")

# Prediction button and logic
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_clicked = st.button('Predict', use_container_width=True)

if predict_clicked:
    if not input_sms.strip():
        st.warning("⚠️ Please enter a message to analyze.")
    else:
        with st.spinner('Analyzing message...'):
            # 1. preprocess
            transformed_sms = transform_text(input_sms)
            # 2. vectorize
            vector_input = tfidf.transform([transformed_sms])
            # 3. predict
            result = model.predict(vector_input)[0]
        
        # 4. Display result
        st.subheader("Result")
        if result == 1:
            st.markdown('<div class="result-box spam-result">🚨 This message is classified as SPAM</div>', unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown('<div class="result-box ham-result">✅ This message is NOT SPAM</div>', unsafe_allow_html=True)
            
        # Show processed text (for transparency)
        with st.expander("See processed text"):
            st.write("The app processed your message as:")
            st.code(transformed_sms)

# Footer
st.markdown("""
<div class="footer">
    Made with <span class="heart">&#10084;&#65039;</span> by Ayush &nbsp;|&nbsp; Contact: 9835237626 &nbsp;|&nbsp; 
    <a href="mailto:ayushashush1111@gmail.com">Email: ayushashush1111@gmail.com</a>
</div>
""", unsafe_allow_html=True)

# Mobile-specific optimizations for font scaling
st.markdown("""
<script>
    // Make the page mobile-friendly with smaller font size
    if (/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)) {
        document.querySelector('html').style.fontSize = '14px';
    }
</script>
""", unsafe_allow_html=True)
