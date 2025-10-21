import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')  # Add this line to download punkt_tab

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


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

# Responsive CSS for all devices
st.markdown("""
<style>
    /* Base responsive styles */
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        font-size: 1.1rem;
        cursor: pointer;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #FF2B2B;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .result-box {
        padding: 1.25rem;
        border-radius: 0.5rem;
        margin-top: 1.5rem;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
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
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1E88E5;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
            margin-bottom: 1rem;
        }
        
        .stButton>button {
            padding: 0.6rem 1.2rem;
            font-size: 1rem;
        }
        
        .result-box {
            padding: 1rem;
            font-size: 1.1rem;
        }
        
        .stTextArea textarea {
            min-height: 120px;
        }
    }
    
    /* Tablet responsiveness */
    @media (max-width: 1024px) and (min-width: 769px) {
        .main-header {
            font-size: 2.2rem;
        }
    }
    
    /* Ensure proper text scaling */
    html {
        font-size: 16px;
    }
    
    /* Make expanders more touch-friendly on mobile */
    .streamlit-expanderHeader {
        font-size: 1.1rem;
        padding: 0.75rem;
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
st.markdown("---")
st.markdown("### How it works:")
st.markdown("""
1. Your message is preprocessed (lowercase, tokenized, cleaned)
2. Stopwords and punctuation are removed
3. Words are stemmed to their root form
4. Processed text is converted to numerical features using TF-IDF
5. A trained machine learning model makes the prediction
""")

# Mobile-specific optimizations
st.markdown("""
<script>
    // Make the page mobile-friendly
    if (/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)) {
        document.querySelector('html').style.fontSize = '14px';
    }
</script>
""", unsafe_allow_html=True)