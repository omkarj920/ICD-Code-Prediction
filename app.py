from sklearn.metrics.pairwise import cosine_similarity
import nltk
import joblib
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from textblob import Word
from nltk.stem import PorterStemmer, WordNetLemmatizer


import streamlit as st
st.title("Dr. Artificial Intelligence")
st.subheader('Specialised in ICD codes worldwide and user based treatment')
st.markdown("---")

import nltk

# Download the necessary NLTK resources
nltk.download('punkt')  # For sentence and word tokenization
nltk.download('stopwords') 
def test_input(inputs,W2v,PC,Z1,NUM):
    from sklearn.metrics.pairwise import cosine_similarity
    import nltk
    from nltk.tokenize import word_tokenize,sent_tokenize
    from nltk.corpus import stopwords
    from gensim.models import Word2Vec
    #from keras.preprocessing.text import Tokenizer
    from textblob import Word
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    stop_words = set(stopwords.words('english'))
    
    def vectorizer(Token,W2v1,vector_size=500):
        vector=[W2v1.wv[i] for i in Token if i in W2v1.wv ]
        if vector:
            return np.mean(vector,axis=0) 
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

    A=process_review(inputs)
    A1=A.split()
    A2=vectorizer(A1,W2v,vector_size=500).tolist()
    A3=PC.transform([A2])
    simm=[]
    for i in range(10000):
        row=Z1.iloc[i,1:-2]
        code=Z1.iloc[i,-2]
        Desc=Z1.loc[i,'DESCRIBE']
        Simillarity=cosine_similarity(A3,[row])
        simm.append((Simillarity,code,Desc))
    Final=pd.DataFrame(simm).sort_values(by=0,ascending=False).head(NUM)
    Final.columns=['SIMILLARITY %', 'ICD CODES','SYMPTOMS']
    Final['SIMILLARITY %'] = Final['SIMILLARITY %'].apply(lambda x: np.around(x[0][0]*100),2)

    return Final


W2v = Word2Vec.load("ICDcodesW2V1.model")
PC=joblib.load("PCA_model_ICD.pkl")
Z1=pd.read_csv("ICD_Codes_Vector_Representation.csv")

test_input('Typhoid fever',W2v,PC,Z1,10)


from langchain_groq import ChatGroq
## OPENAI LLMS
llm1=ChatGroq(temperature=0.8,
             groq_api_key='')



import streamlit as st
import pandas as pd

if 'icd_code' not in st.session_state:
    st.session_state.icd_code = None

if 'predicted_code' not in st.session_state:
    st.session_state.predicted_code = None

# Create layout with 4 columns
col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
# First column: ICD Code Prediction and DataFrame display
with col1:
    st.header("ICD Code Prediction")
    
    user_input = st.text_input("Enter your symptoms or medical information:", key="user_input")
    
    # When the "Predict ICD Code" button is clicked
    if st.button("Predict ICD Code"):
        # Call the function to get predicted codes and store it in session state
        st.session_state.predicted_code = test_input(user_input, W2v, PC, Z1, 10)

        # Display the dataframe as a horizontal table
        st.write("Top 10 Predicted ICD Codes and Symptoms:")
        #st.dataframe(st.session_state.predicted_code, use_container_width=True)
    st.dataframe(st.session_state.predicted_code, use_container_width=True)

# Second column: ICD code selector
with col2:
    if st.session_state.predicted_code is not None:
        st.header("Select ICD Code")
        # Select ICD Code from the dataframe's 2nd column (ICD Codes)
        st.session_state.icd_code = st.selectbox("Select ICD Code from Prediction", 
                                                 options=st.session_state.predicted_code['ICD CODES'].unique(),
                                                 index=0)

# Third column: Collect additional information after ICD code selection
with col3:
    if st.session_state.icd_code:
        st.header("Additional Information")

        # Collect additional information for treatment suggestion
        age = st.number_input("Age", min_value=0, key="age")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="gender")
        sugar = st.number_input("Sugar Level", min_value=0, key="sugar")
        additional_info = st.text_input("Additional Medical Info:", key="additional_info")

        if st.button("Get Treatment"):
            # Generate the treatment suggestion based on the selected ICD code and other info
            treatment_prompt = f"Suggest a treatment for ICD code {st.session_state.icd_code} for a {age}-year-old {gender} with sugar level {sugar} and additional info: {additional_info}."
            
            # Assuming llm1.invoke() generates the treatment suggestion (use your own model here)
            response = llm1.invoke(treatment_prompt)
            
            # Display the suggested treatment
            st.session_state.treatment_response = response.content

# Fourth column: Display Treatment Suggestion output
with col4:
    if 'treatment_response' in st.session_state:
        st.header("Treatment Suggestion")
        st.write(f"Suggested Treatment: {st.session_state.treatment_response}")
