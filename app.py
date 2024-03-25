import numpy as np
import matplotlib.pyplot as plt
import random
import re
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from wordcloud import WordCloud, STOPWORDS
from streamlit_extras.colored_header import colored_header


st.set_page_config(
    page_title="Arbitrum Proposal Co-Pilot",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": "https://twitter.com/sageOlamide",
        "About": None
    }
)

text_1 = '<p style="font-family:sans-serif; color:#4d372c; font-size: 20px;">Explore a visual representation of past successful proposals through our Wordcloud feature. Uncover key words and themes that have led to success, providing you with valuable insights to enhance your own proposals.</p>'
text_2 = '<p style="font-family:sans-serif; color:#4d372c; font-size: 20px;">Predict the success of your proposals with precision using our Success Rate Predictor app. Trained on the texts of past successful proposals, this tool leverages data-driven analysis to forecast the outcome of your submissions.</p>'
text_3 = '<p style="font-family:sans-serif; color:#4d372c; font-size: 20px;">Empower yourself with knowledge, insights, and predictive capabilities to navigate the world of grant proposals confidently. Let Arbitrum Proposal Co-Pilot be your guide to crafting compelling and successful proposals.</p>'
text_4 = '<p style="font-family:sans-serif; color:#4d372c; font-size: 20px;">Dataset and full code can be found in this <a href="https://github.com/OOlajide/arb-proposal-co-pilot/">Github repo</a>.</p>'

st.markdown(f'<h1 style="color:#434346;font-size:60px;text-align:center;">{"Arbitrum Proposal Co-Pilot"}</h1>', unsafe_allow_html=True)
st.markdown(f'<h1 style="color:#434346;font-size:30px;text-align:center;">{"Welcome to Arbitrum Proposal Co-Pilot! 🚀"}</h1>', unsafe_allow_html=True)

st.markdown(f'<h1 style="color:#434346;font-size:30px;text-align:left;">{"Wordcloud Insights"}</h1>', unsafe_allow_html=True)
st.markdown(text_1, unsafe_allow_html=True)
st.markdown(f'<h1 style="color:#434346;font-size:30px;text-align:left;">{"Proposal Success Rate Predictor"}</h1>', unsafe_allow_html=True)
st.markdown(text_2, unsafe_allow_html=True)
st.markdown(text_3, unsafe_allow_html=True)

colored_header(
    label="",
    description="",
    color_name="gray-70",
)
st.markdown(text_4, unsafe_allow_html=True)
colored_header(
    label="",
    description="",
    color_name="gray-70",
)

df = pd.read_csv('grant_dataset.csv')

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader('Proposal Text Wordcloud')
    options = ['All Grant Types']
    options.extend(list(df['grant_name'].unique()))
    # remove grant types with 2 proposals only or less
    options_to_remove = ["Plurality Labs - Firestarters", "Plurality Labs - RFP STIP Monitoring"]
    options = list(filter(lambda x: x not in options_to_remove, options))
    option = st.selectbox(
        'Select a grant type',
        options)
    if option == 'All Grant Types':
        wc_df = df
    else:
        wc_df = df[df['grant_name']==option]
        
    def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)
    text = ' '.join(wc_df['proposal_text'].astype(str).tolist())
    text = text.replace("\n", " ")
    def remove_n_before_capital(text):
        return re.sub(r'n([A-Z])', r'\1', text)
    text = remove_n_before_capital(text)
    # adding stopwords
    stopwords = set(STOPWORDS)
    lst = ['Arbitrum', 'will', 'n', 's', 'ARB', 'nThe','project', 'grant', 'grants', 'https']
    for x in lst:
        stopwords.add(x)
    wc = WordCloud(max_words=1000, stopwords=stopwords, margin=10, random_state=1, width=1000, height=400).generate(text)
    plt.title(f"{option} Proposal Text Word Cloud")
    plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3), interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt.gcf())

with col2:
    # Load your dataset
    data = df.copy()
    data['success_label'] = 1
    # Fill the last 5 rows with 0, these are 5 bad proposals that would likely be rejected
    # Helps the model distinguish a good proposal from a bad one
    data.loc[data.index[-5:], 'success_label'] = 0

    # Feature engineering
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(data['proposal_text'])
    y = data['success_label']

    # Model training using TensorFlow
    X_train, X_test, y_train, y_test = train_test_split(X.toarray(), y, test_size=0.2, random_state=42)
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    st.subheader('Proposal Success Rate Predictor')
    st.info('Proposals with predicted success rate below 80% would likely be rejected.', icon="ℹ️")
    user_input = st.text_area('Enter your proposal text here:', height=300)
    if st.button('Predict Success Rate'):
        user_input_tfidf = tfidf.transform([user_input]).toarray()
        prediction = model.predict(user_input_tfidf)
        success_percentage = round(prediction[0][0] * 100, 2)
        st.write(f'The predicted success rate for your proposal is: {success_percentage}%')
