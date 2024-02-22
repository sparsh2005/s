import numpy as np
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():  # chec`ks if character is alpha-numeric
            y.append(i)
    
    text = [word for word in y if word not in stopwords.words('english') and word not in string.punctuation]
    
    text = [ps.stem(word) for word in text]
    
    return " ".join(text)

# Load your trained vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit code to get user input
st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess the input text
    transformed_sms = transform_text(input_sms)
    
    # 2. Vectorize the preprocessed text
    vector_input = tfidf.transform([transformed_sms]).toarray()
    
    # 3. Append the 'num_characters' feature to the vector_input
    num_characters = len(input_sms)
    vector_input = np.hstack((vector_input, [[num_characters]]))

    # 4. Predict using the model
    result = model.predict(vector_input)[0]
    
    # 5. Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")