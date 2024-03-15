import re
from html import unescape

import contractions
import emoji
import spacy

nlp = spacy.load('data/parser')
stop_words = nlp.Defaults.stop_words


def preprocess(text: str) -> str:
    text = re.sub(r'[^\x00-\x7f]', ' ', text)  # remove non_ascii
    text = unescape(text)  # decode HTML-tags
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)  # remove repetitions
    text = re.sub(r"@\w*", '', text)  # remove user tags
    text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                  '', text)  # remove links
    # remove emojis
    text = re.sub(r"((?::|;|=)(?:-)?(?:\)|D|P|O|\\|/|\())", '', text)
    text = emoji.demojize(text)
    text = re.sub(r"\d{4}s?", '', text) # remove dates
    text = re.sub(r"\b\d[\d\.,/]*(st|th|rd|nd)?\b", '', text)  # remove numerals
    text = re.sub(r'pic\.twitter\.com/\S+', '', text)  # remove pic links
    text = contractions.fix(text)  # fix contractions
    text = ' '.join([token.lemma_ for token in nlp(text) if token.lemma_ not in stop_words])  # lemmatize
    text = text.lower()  # lowercase
    text = ' '.join(re.findall(r'\b[a-z]+\b', text))  # leave only eng text
    return text
