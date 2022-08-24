#!/usr/bin/env python
# coding: utf-8

# In[7]:


#This notebook implements all the necessary functions
#for SC-WEAT-WEAT analyses.


# In[19]:


from gensim.scripts.glove2word2vec import glove2word2vec

#we run it only once, it takes around half an hour. We do this part to convert common crawl vectors into the format we can use.
#aftre that we store the result as a model that we can use for common crawl
#glove2word2vec(glove_input_file='glove.42B.300d.txt', word2vec_output_file="gensim_glove_vectors.txt")

from gensim.models.keyedvectors import KeyedVectors
common_crawl2 = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)


# In[39]:


#This part of the notebook downloads and uploads all the word embedding models
#if you receive an error such as: ... not found, please try to install the library
#by following pip install ...
import math
from scipy import spatial
import gensim.downloader
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#import streamlit as st
import plotly.graph_objs as go
from gensim.models import KeyedVectors

#this is the first common_crawl we worked on, it didnt give correct results
#common_crawl = KeyedVectors.load_word2vec_format('crawl-300d-2M.vec', binary=False)
glove_vectors_google_news = gensim.downloader.load('word2vec-google-news-300')
glove_vectors_twitter = gensim.downloader.load('glove-twitter-200')
glove_vectors_wiki = gensim.downloader.load('glove-wiki-gigaword-100')
glove_vectors_wiki3 = gensim.downloader.load('glove-wiki-gigaword-300')
coha_model_1960 = KeyedVectors.load_word2vec_format('1990.txt', binary=False, unicode_errors='replace')

#structure of the coha pretrained word embeddings are different than the other embeddings,
#it requires some manual preprocessing to be able to reach the embeddings from words

coha_corpus_1960_list  = []
for idx, val in enumerate(coha_model_1960.index_to_key):
    just_word = val.split('_')
    word_key = just_word[0]
    coha_corpus_1960_list.append(word_key)

coha_list = coha_corpus_1960_list
coha_model = coha_model_1960

#####
male_group_words = ['he', 'son', 'his', 'him', 'father', 'man', 'boy', 'himself', 'male', 'brother', 'sons', 'fathers',
                    'men', 'boys', 'males', 'brothers', 'uncle',
                    'uncles', 'nephew', 'nephews']
female_group_words = ["she", 'daughter', 'hers', 'her', 'mother', 'woman', 'girl', 'herself', 'female', 'sister',
                      'daughters', 'mothers', 'women',
                      'girls', 'femen', 'sisters', 'aunt', 'aunts', 'niece', 'nieces']

#change this parameter according to the model you want to use for calculations, putting
#a '#' symbol in front of the line, skips that line, for example, if we want 
#to use twitter embeddings we should put # in front of every other embedding = ... line

embedding = glove_vectors_twitter
#embedding = glove_vectors_google_news
#embedding = glove_vectors_wiki
#embedding = common_crawl2
#embedding = coha_model_1960


# In[40]:


#if a word in the word list doesnt exist in the provided embedding,
#we just skip that word, this function implements that

def collect_existing_words(word_list, embedding):
    lst = []
    if embedding == coha_model:
        for words in word_list:
            existed_words = []
            for word in words:
                if word in coha_list:
                    idx = coha_list.index(word)
                    existed_words.append(embedding.index_to_key[idx])
                else:
                    pass
            lst.append(existed_words)
    else:
        for words in word_list:
            existed_words = []
            for word in words:
                if word in embedding.key_to_index:
                    existed_words.append(word)
                else:
                    pass
            lst.append(existed_words)
    res = list(filter(None, lst))
    return res


# In[41]:


#this function calculates the cosine similarity between two vectors
def cossim(v1, v2, signed = True):
    c = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    if not signed:
        return abs(c)
    return c


# In[42]:


#function to remove stop words from a word phrase before we collect its embedding
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
 
# remove stopwords function
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return filtered_text


# In[43]:


##removing stop words from the given word list, more generalized version of
#remove_Stopwords function

def clean_data(word_list):
    clean_word_list = []
    for wrds in word_list:
        filtered_text = remove_stopwords(wrds)
        clean_word_list.append(filtered_text)
        
    return clean_word_list


# In[44]:


#collecting words' corresponding word embeddings and putting them in an array,
#each row of the array represents one word embedding

def collect_word_embeds(word_list,embedding):
    word_vec = []
    lst = collect_existing_words(word_list, embedding)
    for words in lst:
        vec = 0
        for word in words:
            vect = embedding[word]
            vec += vect
        vect2 = vec / len(words)
        word_vec.append(vect2)
        
    return word_vec


# In[45]:


#another approach: this time we have the list of personality traits from different sources
#(for example one is http://ideonomy.mit.edu/essays/traits.html),
#we calculate their similarity with entrepreneurship according to cosine similarity
#and take the words that are more similar

#this function returns the words that are most similar(top n) and their corresponding
#word embeddings

def collect_predefined_list_embeddings(wordlist_file, embedding, entrepreneurship_word, top_n):
    words = []
    with open(wordlist_file) as f:
        for line in f:
            words.append(line[:-1].lower())
            
    word_dict = {}
    for word in words:
        if word in embedding.key_to_index:
            word_dict[word] = embedding.similarity(entrepreneurship_word, word)
    word_dict2 = sorted(word_dict.items(), key=lambda x:x[1], reverse = True)
    new_words = []
    for word,score in word_dict2[:top_n]:
        new_words.append(word)
    word_embeds = collect_word_embeds(new_words, embedding)
    return new_words, word_embeds


# In[46]:


#calculating sc-weat
#X: female group words
#Y: male group words
#A: attribute group words
#w: one word embedding

#this function calculates the cosine similarity between a given word vector
#and each attribute word in attribute word list, and returns the avg of the
#results. For example we have the word she as one word from group words (w), and 
#we have the home attribute word list(A). What we do is calculating the similarity 
#between she and each word in the attribute list(A) and returning the average of
#the calculation. Result is the similarity between the word she(w), and the attribute
#home(which is a list of words, A)

def s(w, A):
    #the lines below calculates the cosine similarity of the for w for each element
    #(a) of the attribute word list(A), then returns the mean of this calculation as
    #result.
    
    a_cos = np.array([cossim(w, a) for a in A])
    return (np.mean(a_cos))

#This function calculates the SC-WEAT score by using the function s(w,A). X 
#represents all female group words. We use the function s to calculate
#the cosine similarity for each word(x) of the female group list(X). 
#We do the same thing for each word(y) of the male group list(Y).
#SC-weat score is calculated by substracting the average results(female - male)
#then we divide it to standard deviation to return final SC-weat score.

def sc_weat_effect_size(X, Y, A):
    x_s = np.array([s(x, A) for x in X])
    y_s = np.array([s(y, A) for y in Y])

    return (np.mean(x_s) - np.mean(y_s)) / np.std(np.concatenate((x_s, y_s)))




# In[47]:


#these functions below, uses the weight approach introduced in papers but since
#our group word list mostly have the same number of the elements(especially when
#we collect top 50 closest words to represent female or male, both lists has 50
#elements, there is no difference between sc_weat_effect_size method in terms
#of results.)

def weighted_std(values, weights):
	"""
	Return the weighted standard deviation.

	values, weights -- Numpy ndarrays with the same shape.
	"""
	average = np.average(values, weights=weights)
	# Fast and numerically precise:
	variance = np.average((values-average)**2, weights=weights)
	# Small sample size bias correction:
	variance_ddof1 = variance*len(values)/(len(values)-1)
	return math.sqrt(variance_ddof1)

def diff_sim(X, A, B):

		sum_A = 0
		sum_B = 0

		all_sims = []
		for a in A:
			a_ = a.reshape(1, -1)
			results = spatial.distance.cdist(a_, X, 'cosine')
			sum_X = (1 - results).sum()
			val = sum_X/len(X)
			sum_A += val
			all_sims.append(val)
		ave_A = sum_A/len(A)

		for b in B:
			b_ = b.reshape(1, -1)
			results = spatial.distance.cdist(b_, X, 'cosine')
			sum_X = (1 - results).sum()
			val = sum_X/len(X)
			sum_B += val
			all_sims.append(val)
		ave_B = sum_B/len(B)

		difference = ave_A - ave_B

		# For SD calculation, assign weights based on frequency of opposite category
		weights = [len(B) for num in range(len(A))] + [len(A) for num in range(len(B))]
		standard_dev = weighted_std(all_sims, weights)
		effect_size = difference/standard_dev

		return effect_size


# In[48]:


## functions for calculating the p_value

import random
def within_group_cohesion(X):
	dist = spatial.distance.pdist(X, 'cosine')
	return dist.mean()

def group_cohesion_test(X, Y, A,perm_n = 1000):
    
    #test_statistic = sc_weat_effect_size(X, Y, A)
    test_statistic = diff_sim(A, X, Y)
    jointlist = np.concatenate((X,Y))
    permutations = np.array([])

    count = 0
    cutpoint = len(X)
    cutpoint2 = len(Y)
    while count < perm_n:
        np.random.shuffle(jointlist)
        set1 = jointlist[:cutpoint]
        set2 = jointlist[cutpoint:]

        permutations = np.append(permutations, 
                                 diff_sim(A, set1, set2)
                                 #sc_weat_effect_size(set1, set2, A)
        )
        count += 1
   

    P_val = (sum(i <= test_statistic for i in permutations)+1)/(len(permutations)+1)
    
    perm_mean = np.mean(permutations)
    perm_std = np.std(permutations)
    t = (perm_mean - test_statistic) / (perm_std/np.sqrt(perm_n))
    z = (test_statistic - perm_mean) / (perm_std)
    permutations = permutations - perm_mean
    sum_c = test_statistic - perm_mean
    Pleft = (sum(i <= sum_c for i in permutations)+1)/(len(permutations)+1)
    Pright = (sum(i >= sum_c for i in permutations)+1)/(len(permutations)+1)
    Ptot = (sum(abs(i) >= abs(sum_c) for i in permutations)+1)/(len(permutations)+1)
    se = np.std(permutations)
    return P_val, Ptot, z


# In[80]:


#change this parameter according to the model you want to use for calculations, putting
#a '#' symbol in front of the line, skips that line, for example, if we want 
#to use twitter embeddings we should put # in front of every other embedding = ... line

embedding = common_crawl2
#embedding = glove_vectors_google_news
#embedding = glove_vectors_wiki
#embedding = common_crawl
#embedding = glove_vectors_twitter
#embedding = glove_vectors_wiki3


# In[107]:


#here we collect top 50 closest words for female and male to represent female 
#and male as group word lists, top 50 closest words for entrepreneurship attr.
#list

female = []
for word_tuple in embedding.most_similar('female', topn=50):
        female.append(word_tuple[0])
        
male = []
for word_tuple in embedding.most_similar('male', topn=50):
        male.append(word_tuple[0])
        
entrepreneurship = []
for word_tuple in embedding.most_similar('entrepreneurship', topn=50):
        entrepreneurship.append(word_tuple[0])


# In[108]:


entrepreneurship


# In[109]:


#then we clean our group word lists and attr word list to remove stopwords
female_group_words2 = clean_data(female)
male_group_words2 = clean_data(male)
entr_words = clean_data(entrepreneurship)

#and we collect word embeddings for this group word lists and attr. word list
female_word_vectors = collect_word_embeds(female_group_words2, embedding)
male_word_vectors = collect_word_embeds(male_group_words2, embedding)
entr_word_vectors = collect_word_embeds(entr_words, embedding)


# In[110]:


sc_weat_score = sc_weat_effect_size(female_word_vectors, male_word_vectors, entr_word_vectors)
sc_weat_score


# In[111]:


sc_weat_score2 = diff_sim(entr_word_vectors,female_word_vectors,male_word_vectors)
sc_weat_score2


# In[112]:


p, ptot,z = group_cohesion_test(female_word_vectors, male_word_vectors, entr_word_vectors)


# In[113]:


ptot


# In[114]:


#personality traits analysis:
#this part of the analysis focuses on another approach:
# we have the list of personality traits from different sources
#(one is http://ideonomy.mit.edu/essays/traits.html),
#we calculate their similarity with entrepreneurship according to cosine similarity
#and take the words that are more similar
def collect_predefined_list_embeddings(wordlist_file, embedding, entrepreneurship_word, top_n):
    words = []
    with open(wordlist_file) as f:
        for line in f:
            words.append(line[:-1].lower())
            
    word_dict = {}
    for word in words:
        if word in embedding.key_to_index:
            word_dict[word] = embedding.similarity(entrepreneurship_word, word)
    word_dict2 = sorted(word_dict.items(), key=lambda x:x[1], reverse = True)
    new_words = []
    for word,score in word_dict2[:top_n]:
        new_words.append(word)
    word_embeds = collect_word_embeds(new_words, embedding)
    return new_words, word_embeds


# In[127]:


most_similar_words, word_embeds = collect_predefined_list_embeddings("williams-best traits.txt", embedding, 'entrepreneurship', top_n = 50)


# In[128]:


sc__weat_score = sc_weat_effect_size(female_word_vectors, male_word_vectors, word_embeds)


# In[129]:


sc__weat_score


# In[130]:


p, ptot,z = group_cohesion_test(female_word_vectors, male_word_vectors, word_embeds)


# In[131]:


p


# In[132]:


ptot


# In[ ]:




