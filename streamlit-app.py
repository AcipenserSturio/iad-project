import streamlit as st
import pandas as pd
import pickle


loaded_model = pickle.load(open('data/test-model.pkl', 'rb'))
loaded_decoder = pickle.load(open('data/decoder.pkl', 'rb'))


def update_result(text):

    result = loaded_model.predict([text])
    # st.write(loaded_decoder.classes_)

    # st.write(result)
    st.subheader("Predicted cyberbullying class:")
    st.header(loaded_decoder.inverse_transform(result)[0])

    # st.write(loaded_model.predict_log_proba(text))


def on_change_source():
    st.write("huh")


st.title("Detect and classify cyberbullying")

# X_test = ["Good morning, you lovely people <3"]

form = st.form(key='main_form')
source = form.text_input(label="Enter text to predict cyberbullying:")
submit_button = form.form_submit_button(label='Submit')

if submit_button:
    update_result(source)
