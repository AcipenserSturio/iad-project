import streamlit as st
import pandas as pd
import re
import emoji
import sklearn

st.title("Hello world")

st.header("Load Dataset")
df = pd.read_csv('data/cyberbullying_tweets.csv').rename(
    {
        'tweet_text': 'text',
        'cyberbullying_type': 'label'
        },
    axis=1
    )

st.dataframe(df)

st.header("Basic EDA")
st.subheader("Overlook")
# df.info()
# doesn't work for some reason

st.write(df.describe())

st.write(df.sample(15, random_state=0))

st.subheader("Labels Distribution")

import matplotlib.pyplot as plt

fig, axs = plt.subplots(figsize=(10, 5))
df.label.value_counts().plot.pie(autopct="%.2f", ax=axs)
axs.set_title("Classes Balance")

st.pyplot(fig)

labels_quant = df['label'].value_counts()
labels_percent = [f'{round(_, 3) * 100}%' for _ in df['label'].value_counts(normalize=True)]
st.dataframe({'quantity': labels_quant, 'percentage': labels_percent})

st.write(df.label.value_counts())

st.header("Text lengths")
texts = df['text']
st.write(texts.shape)

st.subheader("Average text length")

st.caption("In symbols")
df['text_len'] = [len(text) for text in df['text']]

fig, ax = plt.subplots()
ax.hist(df['text_len'], rwidth=0.9)
st.pyplot(fig)

st.caption("Mean length in symbols")
st.write(round(sum(len(text) for text in texts) / len(texts), 2))

st.subheader("In tokens")
df['tokenized_text_len'] = [len(text.split()) for text in df['text']]
fig, ax = plt.subplots()
ax.hist(df['tokenized_text_len'], rwidth=0.9)
st.pyplot(fig)

st.caption("Mean length in tokens")
st.write(round(sum(len(text.split()) for text in texts) / len(texts), 2))

st.write("Мы видим большие значения на осях X графиков, что может свидетельствовать о наличии аномалий в датасете. Проверим самые длинные и короткие тексты в датасете.")

st.header("Longest and Shortest texts")


longest_text = max(texts, key=len)
left_column, right_column = st.columns(2)
with left_column:
    st.subheader("Longest text")
    st.write(longest_text)
with right_column:
    st.subheader("Symbols count")
    st.write(len(longest_text))
    st.subheader("Words count")
    st.write(len(longest_text.split()))


st.write(df[df['text'].str.contains("is feminazi an actual word with a denot")])

df = df[df['text'].str.contains("\r\n")==False]
df = df.reset_index(drop=True)
texts = df['text']
st.write(texts.shape)

st.write("""Видим наличие аномально большого текста. Судя по отсутствию контекстуальной связи между его частями, этот текст был ошибочно добавлен в датасет: либо из-за несовершенства парсера, обрабатывавшего собранные микросообщения, либо из-за возникшей во время парсинга ошибки. Это подтверждается также разделом "обсуждение" на сайте Kaggle, где и был выложен данный датасет -- там также пришли ко мнению о том, что это ошибка.

Дальнейшее изучение текстов показало, что такой случай не единичный. В итоге нами было принято решение удалить такие тексты из датасета. Мы аргументируем наш выбор следующими доводами: во-первых, попытка разделить и разметить проблемные тексты отняла бы много времени, поскольку такой случай не единичный, и в итоге текстов, которые должны быть размечены в отдельности, набирается довольно много. Кроме того, у нас нет возможности проверить правильность возможной разметки, таким образом, неправильная разметка может сказаться на качестве работы модели. В случае же, если бы мы просто разделили эти тексты и добавили в датасет неразмеченными, от них было бы мало смысла, так как их нельзя было бы использовать ни в обучении, ни в валидации модели по причине отсутствия разметки.

Для удаления больших текстов мы решили опираться на паттерн, а не длину сообщений, поскольку на неё влияют добавленные в сообщение ссылки, смайлики и проч., увеличивающие реальную длину сообщения в символах. Мы выявили, что ошибочные тексты содержат сочетание символов \\r\\n и удалили все тексты, содержащие данное сочетание символов.
""")

longest_text = max(texts, key=len)
left_column, right_column = st.columns(2)
with left_column:
    st.subheader("Longest text (corrected)")
    st.write(longest_text)
with right_column:
    st.subheader("Symbols count")
    st.write(len(longest_text))
    st.subheader("Words count")
    st.write(len(longest_text.split()))

st.header("Shortest text")

shortest_text = min(texts, key=len)
left_column, right_column = st.columns(2)
with left_column:
    st.subheader("Longest text")
    st.write(shortest_text)
with right_column:
    st.subheader("Symbols count")
    st.write(len(shortest_text))
    st.subheader("Words count")
    st.write(len(shortest_text.split()))

labels_quant = df['label'].value_counts()
labels_percent = [f'{round(_, 3) * 100}%' for _ in df['label'].value_counts(normalize=True)]
st.dataframe({'quantity': labels_quant, 'percentage': labels_percent})



st.subheader("Check average text length again")
st.caption("In symbols")

df['text_len'] = [len(text) for text in df['text']]

fig, ax = plt.subplots()
ax.hist(df['text_len'], rwidth=0.9)
st.pyplot(fig)

st.write(round(sum(len(text) for text in texts) / len(texts), 2))

st.subheader("In tokens")
df['tokenized_text_len'] = [len(text.split()) for text in df['text']]
fig, ax = plt.subplots()
ax.hist(df['tokenized_text_len'], rwidth=0.9)
st.pyplot(fig)

st.write(round(sum(len(text.split()) for text in texts) / len(texts), 2))\

# Clear abundant columns
df.drop('text_len', axis=1, inplace=True)
df.drop('tokenized_text_len', axis=1, inplace=True)
df.head()


st.header("Preprocessing")

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

def tokenize(text: str) -> str:
  return " ".join(re.findall(r"\w*", text))

def lowercase(text: str) -> str:
  return text.lower()

df['filtered_text'] = df['text'] \
    .apply(replace_mentions) \
    .apply(replace_urls) \
    .apply(replace_emotes) \
    .apply(replace_dates) \
    .apply(replace_numerals) \
    .apply(tokenize) \
    .apply(lowercase)

st.write(df.sample(15, random_state=23))


df.drop_duplicates(inplace=True)
df.reset_index(inplace=True, drop=True)
st.write(df.shape)



df.drop_duplicates(inplace=True)
df.reset_index(inplace=True, drop=True)
st.write(df.shape)


# sourced: https://habr.com/ru/articles/538458/
from wordcloud import WordCloud
# Получение текстовой строки из списка слов
def str_corpus(corpus):
    str_corpus = ''
    for i in corpus:
        str_corpus += ' ' + i
    str_corpus = str_corpus.strip()
    return str_corpus
# Получение списка всех слов в корпусе
def get_corpus(data):
    corpus = []
    for phrase in data:
        for word in phrase.split():
            corpus.append(word)
    return corpus
# Получение облака слов
def get_wordCloud(corpus):
    wordCloud = WordCloud(background_color='black',
                              width=3000,
                              height=2500,
                              max_words=200,
                              random_state=42
                         ).generate(str_corpus(corpus))
    return wordCloud

corpus = get_corpus(df['text'].values)
procWordCloud = get_wordCloud(corpus)

fig = plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.imshow(procWordCloud)
plt.axis('off')
plt.subplot(1, 2, 1)
st.pyplot(fig)
