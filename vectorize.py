#!/usr/bin/env python3

import numpy as np
from gensim.models.doc2vec import Doc2Vec
from nltk import word_tokenize
from sklearn import feature_extraction

def remove_all_stopwords(tokens):
    return [w for w in tokens if w not in feature_extraction.text.ENGLISH_STOP_WORDS]

def text_to_vector(model, text):
    text_words = remove_all_stopwords(word_tokenize(text))
    model.random.seed(0)
    text_vector = model.infer_vector(text_words)
    return text_vector

doc_model = Doc2Vec.load('doc2vec.bin')
example = 'This is a string of text.'
vec = text_to_vector(doc_model, example)

print(vec)
