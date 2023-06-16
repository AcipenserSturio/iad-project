import streamlit as st
import pandas as pd
import pickle

st.title("Hello world")

# X_test = ["Good morning, you lovely people <3"]

text = [st.text_area('Input text to predict:')]

loaded_model = pickle.load(open('data/test-model.pkl', 'rb'))
loaded_decoder = pickle.load(open('data/decoder.pkl', 'rb'))

result = loaded_model.predict(text)
st.write(loaded_decoder.classes_)

# st.write(result)
st.write(loaded_decoder.inverse_transform(result))

st.write(loaded_model.predict_log_proba(text))
