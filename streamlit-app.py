import pickle
import re

import emoji
import nltk
import pandas as pd
import streamlit as st
from nltk.stem import WordNetLemmatizer


@st.cache_data
def load_nltk_packs():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')


# loaded_model = pickle.load(open('data/tf_idf_logreg_finetuned.pkl', 'rb'))
# loaded_model_2 = pickle.load(open('data/cvec_lsvc_finetuned.pkl', 'rb'))
loaded_model_new = pickle.load(open('data/tf_idf_logreg_finetuned_new.pkl', 'rb'))
loaded_model_2_new = pickle.load(open('data/cvec_lsvc_finetuned_new.pkl', 'rb'))
loaded_decoder = pickle.load(open('data/decoder.pkl', 'rb'))
load_nltk_packs()


def preprocess(text):
    def replace_mentions(text: str) -> str:
        return re.sub(r"@\w*", 'USER_TAG_PLACEHOLDER', text)

    def replace_urls(text: str) -> str:
        # sourced: https://urlregex.com/
        return re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", 'LINK_PLACEHOLDER', text)

    def replace_emotes(text: str) -> str:
        text = re.sub(r"((?::|;|=)(?:-)?(?:\)|D|P|O|\\|/|\())", 'EMOTE_PLACEHOLDER', text)
        text = emoji.demojize(text)
        return text

    def replace_dates(text: str) -> str:
        return re.sub(r"\d{4}s?", 'DATE_PLACEHOLDER', text)

    def replace_numerals(text: str) -> str:
        return re.sub(r"\b\d[\d\.,/]*(st|th|rd|nd)?\b", 'NUMERAL_PLACEHOLDER', text)

    def replace_repetitions(text: str) -> str:
        return re.subre.sub(r"(.)\1{2,}", r"\1\1", text)

    def lemmatize(text: str) -> str:
        lemmatizer = WordNetLemmatizer()
        return " ".join([lemmatizer.lemmatize(token) for token in re.findall(r"\w*", text)])

    def lowercase(text: str) -> str:
        return text.lower()

    text = replace_mentions(text)
    text = replace_urls(text)
    text = replace_emotes(text)
    text = replace_dates(text)
    text = replace_numerals(text)
    text = lemmatize(text)
    text = lowercase(text)
    return text


def on_change_source():
    st.write("huh")


st.title("Detect and classify cyberbullying")

# X_test = ["Good morning, you lovely people <3"]

model_choice = {
    # "LogReg + TF-IDF": loaded_model,
    # "LinearSVC + CVec": loaded_model_2,
    "[NEW] LogReg + TF-IDF": loaded_model_new,
    "[NEW] LinearSVC + CVec": loaded_model_2_new
}

option = st.selectbox(
    "Model",
    (
        # "LogReg + TF-IDF",
        # "LinearSVC + CVec",
        "[NEW] LogReg + TF-IDF",
        "[NEW] LinearSVC + CVec"
    )
)

form = st.form(key='main_form')
text = form.text_input(label="Enter text to predict cyberbullying:")
submit_button = form.form_submit_button(label='Submit')

if submit_button:

    text = preprocess(text)
    result = model_choice[option].predict([text])
    # st.write(loaded_decoder.classes_)

    # st.write(result)
    st.subheader("Predicted cyberbullying class:")
    st.header(loaded_decoder.inverse_transform(result)[0])

    # st.write(loaded_model.predict_log_proba(text))
