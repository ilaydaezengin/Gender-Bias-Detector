from typing import List

import gensim.downloader
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import streamlit as st
import plotly.graph_objs as go

glove_vectors_google_news = gensim.downloader.load('word2vec-google-news-300')
glove_twitter = gensim.downloader.load('glove-twitter-200')
glove_wiki = gensim.downloader.load('glove-wiki-gigaword-100')
# glove_vectors_twitter = gensim.downloader.load('glove-twitter-200')
# word_vectors = []

male_group_words = ['he', 'son', 'his', 'him', 'father', 'man', 'boy', 'himself', 'male', 'brother', 'sons', 'fathers',
                    'men', 'boys', 'males', 'brothers', 'uncle',
                    'uncles', 'nephew', 'nephews']
female_group_words = ["she", 'daughter', 'hers', 'her', 'mother', 'woman', 'girl', 'herself', 'female', 'sister',
                      'daughters', 'mothers', 'women',
                      'girls', 'femen', 'sisters', 'aunt', 'aunts', 'niece', 'nieces']
job_attr_list = ['statistician', 'auctioneer', 'photographer', 'geologist', 'accountant', 'physicist', 'dentist',
                 'psychologist', 'supervisor', 'mathematician', 'designer', 'economist', 'postmaster', 'broker',
                 'chemist',
                 'librarian', 'scientist', 'instructor',
                 'pilot', 'administrator', 'architect', 'surgeon', 'nurse', 'engineer', 'lawyer', 'physician',
                 'manager',
                 'official',
                 'doctor', 'professor',
                 'student', 'judge', 'teacher', 'author']
outsider_adj_list = ['devious', 'bizarre', 'venomous', 'erratic', 'barbaric', 'frightening', 'deceitful', 'forceful',
                     'deceptive', 'envious', 'greedy', 'hateful', 'contemptible', 'brutal',
                     'monstrous', 'calculating', 'cruel', 'intolerant', 'aggressive', 'monstrous']
competence_adj_list = ['precocious', 'resourceful', 'inquisitive', 'sagacious', 'inventive', 'astute', 'adaptable',
                       'reflective', 'discerning', 'intuitive',
                       'inquiring', 'judicious', 'analytical', 'luminous', 'venerable', 'imaginative', 'shrewd',
                       'thoughtful', 'sage',
                       'smart', 'ingenious', 'clever', 'brilliant', 'logical', 'intelligent', 'apt', 'genius', 'wise']
terrorism_related_list = ['terror', 'terrorism', 'violence', 'attack', 'death', 'military', 'war', 'radical', 'injuries', 'bomb', 'target',
'conflict', 'dangerous', 'kill', 'murder', 'strike', 'dead', 'violence', 'fight', 'death', 'force', 'stronghold', 'wreckage', 'aggression',
'slaughter', 'execute', 'overthrow', 'casualties', 'massacre', 'retaliation', 'proliferation', 'militia', 'hostility', 'debris', 'acid',
'execution', 'militant', 'rocket', 'guerrilla', 'sacrifice', 'enemy', 'soldier', 'terrorist', 'missile', 'hostile', 'revolution', 'resistance',
'shoot']
psyc_appearance_list = ['alluring', 'voluptuous', 'blushing', 'homely', 'plump', 'sensual', 'gorgeous', 'slim', 'bald', 'athletic', 'fashionable',
                       'stout', 'ugly', 'muscular', 'slender', 'feeble', 'handsome', 'healthy', 'attractive', 'fat', 'weak', 'thin', 'pretty',
'beautiful', 'strong']
occupations_list = ['janitor', 'statistician', 'midwife', 'bailiff', 'auctioneer', 'photographer', 'geologist', 'shoemaker', 'athlete', 'cashier',
'dancer', 'housekeeper', 'accountant', 'physicist', 'gardener', 'dentist', 'weaver', 'blacksmith', 'psychologist', 'supervisor',
'mathematician', 'surveyor', 'tailor', 'designer', 'economist', 'mechanic', 'laborer', 'postmaster', 'broker', 'chemist', 'librarian', 'attendant', 'clerical', 'musician', 'porter',
                    'scientist', 'carpenter', 'sailor', 'instructor', 'sheriff', 'pilot', 'inspector', 'mason',
'baker', 'administrator', 'architect', 'collector', 'operator', 'surgeon', 'driver', 'painter', 'conductor', 'nurse', 'cook', 'engineer',
'retired', 'sales', 'lawyer', 'clergy', 'physician', 'farmer', 'clerk', 'manager', 'guard', 'artist', 'smith', 'official', 'police', 'doctor',
'professor', 'student', 'judge', 'teacher', 'author', 'secretary', 'soldier']
attr_list = []




# word_vectors = [female_group_vector[0], male_group_vector[0]]


def app():
    st.header('Gender Bias in Pretrained Word Embeddings')
    st.write(
        'Word embeddings are vector representations of words. Words with similar meanings are closer together in vector space.')
    st.write('Recent works demonstrate that word embeddings capture common stereotypes because'
             ' these stereotypes are likely to be present, even if subtly in the large corpora of training texts.')
    st.write(
        'This page shows an interactive analysis between pretrained word embeddings and stereotypes in different domains.')

    option = st.selectbox('Please select a pretrained word embedding.', ('Glove-Google News', 'Glove-Wikipedia', 'Glove-Twitter'))
    option_field = st.selectbox('Please choose the domain to see the bias in that domain.',
                                ('professional occupations', 'competence adjectives', 'outsider adjectives', 'terrorism related words', 'physical appearance adjectives','occupations'))

    vis_option = st.selectbox('Please select visualization dimension.',('2D','3D'))

    if option== 'Glove-Google News':
        model = glove_vectors_google_news
    elif option == 'Glove-Wikipedia':
        model = glove_wiki
    else:
        model = glove_twitter


    if option_field == 'professional occupations':
        word_vectors = []
        attr_list = job_attr_list
        for word in attr_list:
            if word in model.key_to_index.keys():
                vect = model[word]
                word_vectors.append(vect)
            else:
                pass

    elif option_field == 'outsider adjectives':
        word_vectors = []
        attr_list = outsider_adj_list
        for word in attr_list:
            if word in model.key_to_index.keys():
                vect = model[word]
                word_vectors.append(vect)
            else:
                pass

    elif option_field == 'terrorism related words':
        word_vectors = []
        attr_list = terrorism_related_list
        for word in attr_list:
            if word in model.key_to_index.keys():
                vect = model[word]
                word_vectors.append(vect)
            else:
                pass

    elif option_field == 'physical appearance adjectives':
        word_vectors = []
        attr_list = psyc_appearance_list
        for word in attr_list:
            if word in model.key_to_index.keys():
                vect = model[word]
                word_vectors.append(vect)
            else:
                pass

    elif option_field == 'occupations':
        word_vectors = []
        attr_list = occupations_list
        for word in attr_list:
            if word in model.key_to_index.keys():
                vect = model[word]
                word_vectors.append(vect)
            else:
                pass

    else:
        word_vectors = []
        attr_list = competence_adj_list
        for word in attr_list:
            if word in model.key_to_index.keys():
                vect = model[word]
                word_vectors.append(vect)
            else:
                pass

    def create_female_group_vec():
        female_group_vec = []
        count = 0
        sum_words = 0
        for wrd in female_group_words:
            if wrd in model.key_to_index.keys():
                count += 1
                vec = model[wrd]
                sum_words += vec
            else:
                pass

        female_group_vec.append(sum_words / count)
        return female_group_vec

    def create_male_group_vec():
        male_group_vec = []
        count = 0
        sum_words = 0
        for wrd in male_group_words:
            if wrd in model.key_to_index:
                count += 1
                vec = model[wrd]
                sum_words += vec
            else:
                pass

        male_group_vec.append(sum_words / count)
        return male_group_vec

    def cossim(v1, v2, signed=True):
        c = np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)
        if not signed:
            return abs(c)
        return c

    def calc_bias(female_group_vector, male_group_vector, word_vectors):
        arr = 0
        for vect in word_vectors:
            a = cossim(female_group_vector, vect)
            arr += a
        female_bias = arr / len(word_vectors)

        arr2 = 0
        for vect in word_vectors:
            a = cossim(male_group_vector, vect)
            arr2 += a
        male_bias = arr2 / len(word_vectors)

        return female_bias[0] - male_bias[0]

    def calc_bias2(female_group_vector, male_group_vector, word_vectors):
        arr = 0
        for vect in word_vectors:
            a = cossim(female_group_vector, vect)
            a2 = cossim(male_group_vector, vect)
            arr += a2-a
        bias = arr / len(word_vectors)


        return bias

    female_group_vector = create_female_group_vec()
    male_group_vector = create_male_group_vec()

    def Visualize_2D(word_vectors):
        tsne = TSNE(n_components=2, random_state=42, n_iter=5000, perplexity=5)
        np.set_printoptions(suppress=True)
        word_vectors = np.vstack([word_vectors, female_group_vector])
        word_vectors = np.vstack([word_vectors, male_group_vector])
        T = tsne.fit_transform(word_vectors)[:, :3]
        labels = attr_list + ['female', 'male']

        fig = plt.figure(figsize=(12, 6))
        plt.scatter(T[:, 0], T[:, 1], c='orange', edgecolors='r')
        for label, x, y in zip(labels, T[:, 0], T[:, 1]):
            plt.annotate(label, xy=(x + 1, y + 1), xytext=(0, 0), textcoords='offset points')

        st.pyplot(fig)

    bias = calc_bias(female_group_vector, male_group_vector, word_vectors)
    bias2 = calc_bias2(female_group_vector, male_group_vector, word_vectors)
    st.markdown(
        'Gender bias for the attribute words in the domain ' + option_field + ' for the ' + option + ' pretrained word '
                                                                                                     'embeddings is '
        + str(bias) + '.')
    if bias < 0:
        st.markdown('Bias is negative. This means ' + option_field + ' domain is biased towards male gender.')
    else:
        st.markdown('Bias is positive. This means ' + option_field + ' domain is biased towards female gender.')

    st.markdown('Gender bias2 is: ' + str(bias2) )



    def display_pca_scatterplot_3D(model=glove_vectors_google_news, user_input=None, words=None, label=None,
                                   color_map=None,
                                   sample=10, word_vectors=None):
        if words is None:
            if sample > 0:
                words = np.random.choice(list(model.vocab.keys()), sample)
            else:
                words = [word for word in model.vocab]

        # word_vectors = word_vectors.append(female_group_vector)
        # word_vectors = word_vectors.append(male_group_vector)

        word_vectors = np.array([model[w] for w in words])
        word_vectors = np.vstack([word_vectors, female_group_vector])
        word_vectors = np.vstack([word_vectors, male_group_vector])
        three_dim = PCA(random_state=0).fit_transform(word_vectors)[:, :3]

        data = []
        count = 0

        '''for i in range(len(user_input)):
            trace = go.Scatter3d(
                x=three_dim[count:count + topn, 0],
                y=three_dim[count:count + topn, 1],
                z=three_dim[count:count + topn, 2],
                text=words[count:count + topn],
                name=user_input[i],
                textposition="top center",
                textfont_size=20,
                mode='markers+text',
                marker={
                    'size': 10,
                    'opacity': 0.8,
                    'color': 2
                }
    
            )
    
            data.append(trace)
            #count = count + topn'''

        trace_input = go.Scatter3d(
            x=three_dim[count:, 0],
            y=three_dim[count:, 1],
            z=three_dim[count:, 2],
            text=words[count:],
            name='input words',
            textposition="top center",
            textfont_size=20,
            mode='markers+text',
            marker={
                'size': 10,
                'opacity': 1,
                'color': 'black'
            }
        )

        data.append(trace_input)

        layout = go.Layout(
            margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
            showlegend=False,
            legend=dict(
                x=1,
                y=0.5,
                font=dict(
                    family="Courier New",
                    size=25,
                    color="black"
                )),
            font=dict(
                family=" Courier New ",
                size=15),
            autosize=False,
            width=700,
            height=700
        )

        plot_figure = go.Figure(data=data, layout=layout)
        st.plotly_chart(plot_figure)

    labels = attr_list + ['female', 'male']
    label_dict = dict([(y, x + 1) for x, y in enumerate(set(labels))])
    color_map = [label_dict[x] for x in labels]

    if vis_option == '2D':
        Visualize_2D(word_vectors)
    else:
        try: display_pca_scatterplot_3D(glove_vectors_google_news, attr_list, labels, color_map)
        except NameError:
            pass

# print(size(np.array(glove_vectors_google_news['she'])))
# print(size(female_group_vector))


'''


def calc_distance_between_vectors(vec1, vec2, distype = 'norm'):
    if distype == 'norm':
        return np.linalg.norm(np.subtract(vec1, vec2))
    else:
        return cossim(vec1, vec2)'''
