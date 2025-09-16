# 📱 SMS Spam Classifier  

A **machine learning powered web app** built with **Streamlit** that classifies SMS messages as **Spam** or **Not Spam** in real-time.  
This project uses **Natural Language Processing (NLP)** techniques like text preprocessing, stopword removal, stemming, and **TF-IDF vectorization** to analyze and classify messages.

## 🚀 Features  

✅ Classifies SMS as **Spam** or **Not Spam**  
✅ Clean and mobile-responsive UI with custom CSS  
✅ Transparent workflow – see how text is preprocessed  
✅ Works in real-time using a trained ML model  
✅ Built with **Streamlit** for easy deployment  

## 🛠️ Tech Stack  

- **Frontend/UI** → [Streamlit](https://streamlit.io/) + Custom CSS  
- **ML/NLP** → Scikit-learn, NLTK, TF-IDF Vectorizer  
- **Model** → Trained classification model (pickle serialized)  
- **Language** → Python 3.8+  


## ⚙️ Installation & Setup  

1. **Clone the repository**  
git clone https://github.com/Ayush-Raj189/SpamGuard-Pro.git
cd SpamGuard-Pro

2.Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

3.Run the app
streamlit run app.py

🔎How It Works
1.Text Preprocessing
   Converts message to lowercase
   Tokenizes words
   Removes stopwords & punctuation
   Applies stemming

2.Feature Extraction
   Transforms text into numerical features using TF-IDF

3.Prediction
   ML model classifies the message as Spam (1) or Not Spam (0)

4.Output
  Displays classification result with styled result box
  Optionally shows the processed text for transparency

📸 Screenshots
<img width="1748" height="1388" alt="image" src="https://github.com/user-attachments/assets/6386f27d-b902-4659-83e0-ee3cfbe1c151" />


🔮 Future Improvements

📌 Add support for multiple languages
📌 Deploy the app on Streamlit Cloud / Heroku / AWS
📌 Improve accuracy with deep learning models (LSTM/BERT)
📌 Build an API for external integration

🤝 Contributing

Contributions are welcome!
Feel free to fork this repo, open issues, and submit pull requests.

👨‍💻 Author
Developed with ❤️ by Ayush
