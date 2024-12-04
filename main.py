import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from textblob import Word
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Load models and data
W2v = Word2Vec.load("ICDcodesW2V1.model")
PC = joblib.load("PCA_model_ICD.pkl")
Z1 = pd.read_csv("ICD_Codes_Vector_Representation.csv")

# Function to process input and predict ICD codes
def test_input(inputs, W2v, PC, Z1, NUM):
    stop_words = set(stopwords.words('english'))

    def vectorizer(Token, W2v1, vector_size=500):
        vector = [W2v1.wv[i] for i in Token if i in W2v1.wv]
        if vector:
            return np.mean(vector, axis=0)
        else:
            return np.zeros(vector_size)

    def process_review(review):
        sentences = sent_tokenize(review)
        processed_words = []
        for sentence in sentences:
            words = word_tokenize(sentence)
            processed_sentence = [
                str(Word(word).lemmatize()) for word in words if word.isalpha() and word.lower() not in stop_words
            ]
            processed_words.extend(processed_sentence)
        return ' '.join(processed_words)

    # Process input and predict similarity
    A = process_review(inputs)
    A1 = A.split()
    A2 = vectorizer(A1, W2v, vector_size=500).tolist()
    A3 = PC.transform([A2])
    
    simm = []
    for i in range(100):
        row = Z1.iloc[i, 1:-2]
        code = Z1.iloc[i, -2]
        Desc = Z1.loc[i, 'DESCRIBE']
        Similarity = cosine_similarity(A3, [row])
        simm.append((Similarity, code, Desc))

    Final = pd.DataFrame(simm).sort_values(by=0, ascending=False).head(NUM)
    Final.columns = ['SIMILLARITY %', 'ICD CODES', 'SYMPTOMS']
    Final['SIMILLARITY %'] = Final['SIMILLARITY %'].apply(lambda x: np.around(x[0][0] * 100), 2)

    return Final

# Streamlit app
st.title("ICD Code Predictor and Treatment Suggestion")
st.markdown("---")

# Initialize session state
if 'icd_code' not in st.session_state:
    st.session_state.icd_code = None
    st.session_state.predicted_df = None

# Left column: ICD code prediction
left, right = st.columns([1, 1])

with left:
    st.header("ICD Code Prediction")
    user_input = st.text_input("Enter your symptoms or medical information:")

    # If user inputs symptoms, get predictions
    if st.button("Predict ICD Code"):
        predicted_code = test_input(user_input, W2v, PC, Z1, 10)
        
        # Store predicted dataframe in session state
        st.session_state.predicted_df = predicted_code
        
        # Display the DataFrame as a table in Streamlit
        st.write("Top 10 Predicted ICD Codes and Symptoms:")
        st.dataframe(predicted_code)  # This will render the dataframe as a table

# Selector for ICD code
with left:
    if st.session_state.predicted_df is not None:
        icd_code = st.selectbox("Select an ICD Code:", st.session_state.predicted_df['ICD CODES'].unique())

        # Store the selected ICD code
        st.session_state.icd_code = icd_code

# Right column: Treatment suggestion (appears when ICD code is selected)
if st.session_state.icd_code is not None:
    with right:
        st.header("Treatment Suggestion")
        age = st.number_input("Age", min_value=0)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        sugar = st.number_input("Sugar Level")
        additional_info = st.text_input("Additional Medical Info:")

        if st.button("Get Treatment"):
            # Integrate LLM for treatment suggestions (replace llm1.invoke with actual LLM integration)
            treatment_prompt = f"Suggest a treatment for ICD code {st.session_state.icd_code} for a {age}-year-old {gender} with sugar level {sugar} and additional info: {additional_info}."
            response = treatment_prompt  # Example, replace with actual LLM call
            st.write(f"Suggested Treatment: {response}")
