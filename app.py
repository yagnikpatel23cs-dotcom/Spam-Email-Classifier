import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Spam Email Classifier", page_icon="🛡️")

@st.cache_resource
def train_real_model():
    try:
        df = pd.read_csv('spam_ham_dataset.csv')
    except FileNotFoundError:
        st.error("Dataset not found! Please ensure 'spam_ham_dataset.csv' is in the folder.")
        return None, None, 0

    X = df['text']
    y = df['label_num'] # 0 = ham, 1 = spam

    #  Split & Vectorize
    vectorizer = CountVectorizer(stop_words='english') # Removes 'the', 'is', etc.
    X_vec = vectorizer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    # Train Model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    
    return vectorizer, model, acc

# Initialize
vectorizer, model, accuracy = train_real_model()

# --- APP 
st.title("🛡️Spam Email Classifier")
st.write(f"Model trained on **5,172 real emails** from the Enron dataset.")
st.write(f"📊 **Model Accuracy:** {accuracy*100:.2f}%")

user_email = st.text_area("Paste email text here:", height=200)

if st.button("Analyze"):
    if user_email:
        transformed_email = vectorizer.transform([user_email])
        prediction = model.predict(transformed_email)[0]
        
        if prediction == 1:
            st.error("🚨 **SPAM DETECTED**")
            st.warning("This email matches patterns found in known junk or phishing mail.")
        else:
            st.success("✅ **SAFE (HAM)**")
            st.info("This looks like a legitimate message.")
    else:
        st.info("Please enter email content to see the prediction.")