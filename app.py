import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load resources
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Load model and vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Preprocess input
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    filtered = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(filtered)

# Streamlit UI
st.title("📧 Spam Message Detector")

user_input = st.text_area("Enter a message to classify:")

if st.button("Predict"):
    cleaned_input = preprocess(user_input)
    vectorized_input = vectorizer.transform([cleaned_input]).toarray()
    prediction = model.predict(vectorized_input)[0]

    if prediction == 1:
        st.error("🚨 This message is **SPAM**.")
    else:
        st.success("✅ This message is **HAM** (Not Spam).")
