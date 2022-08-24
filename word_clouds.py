import pandas as pd
import string
import os
import spacy
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
from spacy import displacy
import regex as re
from sklearn.decomposition import PCA
import streamlit as st
from textblob import TextBlob

path = r'C:/Users/ilayd/IDP/Scripts/headlines/data/'
files = os.listdir(path)

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import requests
filename = 'all_data.csv'
final_df = pd.read_csv(path + filename)
finance_wordlist = ['executive', 'management', 'professional', 'corporation', 'salary', 'office', 'business', 'career']
home_wordlist = ['home', 'parents', 'children', 'family', 'cousins', 'marriage', 'wedding', 'relatives']


def app():
    st.markdown('The goal of word cloud section of the project is capturing and visualizing adjectives from the articles that has the attribute words.'
                ' You can also narrow down the data by choosing a field. Since it is a live analysis, it takes some time to create the word clouds.')
    # Collecting all data in one dataframe
    option = st.selectbox("Choose the attribute words field: ",
                          ('finance', 'home'))
    if option == 'finance':
        lst = finance_wordlist
    else:
        lst = home_wordlist

    option_field = st.selectbox("Choose a field: ",
                                ('arts', 'at-home', 'world', 'sports',
                                 'well', 'us',
                                 'technology', 'science', 'business', 'books', 'briefing', 'climate', 'education',
                                 'garden', 'health', 'insider', 'magazine', 'movies',
                                 'obituaries', 'parenting', 'realestate', 'style', 'theater', 'todayspaper',
                                 'travel',
                                 'dining', 'opinion', 'learning', 'podcasts', 'fashion', 'your-money',
                                 'Include all articles'))

    if option_field == 'Include all articles':
        new_data = final_df
    else:
        new_data = final_df[final_df['web_url'].str.contains(option_field, na=False)]



    # collecting article index
    index_list = []
    for index, row in new_data.iterrows():
        # if any(map(lambda i: i in finance_wordlist, row)):

        if any(check in row['content'] for check in lst):
            index_list.append(index)

    new_data_filtered = new_data.filter(items=index_list, axis=0)

    def get_adjectives(text):
        blob = TextBlob(text)
        return [word for (word, tag) in blob.tags if tag == "JJ"]

    new_data_filtered['adjectives'] = new_data_filtered['content'].apply(get_adjectives)
    # print(new_data_filtered)
    # collecting words to create word cloud
    comment_words = ' '
    stopwords = set(STOPWORDS)
    for val in new_data_filtered.adjectives:
        # print(val)
        tokens = str(val).split()
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        comment_words += " ".join(tokens) + " "

    pic = np.array(Image.open(requests.get("http://www.clker.com/cliparts/O/i/x/Y/q/P/yellow-house-hi.png", stream=True).raw))

    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=stopwords, mask = pic,
                          min_font_size=10).generate(comment_words)


    st.markdown('Number of articles used in this word cloud is ' + len(index_list))
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(wordcloud)
    plt.axis("off")
    st.pyplot(fig)
