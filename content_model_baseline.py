# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:16 2017

@author: nfitch3
"""
from __future__ import unicode_literals, print_function

from sklearn.feature_extraction.text import TfidfVectorizer
import string
import numpy as np
import os
from sklearn import feature_extraction
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
#from imblearn.under_sampling import RandomUnderSampler
#from imblearn.over_sampling import RandomOverSampler
#from sklearn.neural_network import MLPClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from dataset import get_article_text
import csv
#from collections import Counter

import sys
#from split_dataset import generate_hold_out_split

def remove_punctuation(myString):
    translator = str.maketrans('', '', string.punctuation)
    # Remove punctuation marks
    return myString.translate(translator)

def remove_stop_words(tokens):
    return [w for w in tokens if w not in feature_extraction.text.ENGLISH_STOP_WORDS]

def return_top_k_keys(myDict,k):
    # Returns keys with largest k values (i.e. counts)
    return sorted(myDict, key=myDict.get, reverse=True)[:k]

def clean(myString):
    myTokens_no_numbers = []
    myTokens = remove_punctuation(myString.lower()).split()
    myTokens_no_stops = remove_stop_words(myTokens)
    for token in myTokens_no_stops:
        if not any(char.isdigit() for char in token):
            myTokens_no_numbers.append(token)
    
    myString_cleaned = " ".join(myTokens_no_numbers)
    
    return myString_cleaned



 
# Read in text data
file_name = "gdelt_text.csv"
if not (os.path.exists(file_name)):
    #Get articles by domain name
    articles = get_article_text()
    documents = articles.keys()
    with open(file_name, "w",encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["sdom","text"])
        # Concatenate each domain's text into corpus
        corpus = []
        for sdom in documents:
            corpus.append(articles[sdom])
            # Write rows
            writer.writerow([sdom,articles[sdom]])

else:
    # Read in articles from file_name
    maxInt = sys.maxsize
    decrement = True   
    articles = {}
    with open(file_name, 'r',encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            while decrement:
                # decrease the maxInt value by factor 10 
                # as long as the OverflowError occurs.
                decrement = False
                try:
                    csv.field_size_limit(maxInt)
                except OverflowError:
                    maxInt = int(maxInt/10)
                    decrement = True
            if row[0] not in articles:
                articles[row[0]] = []
            articles[row[0]].append(row[1])

# Read in labels and add to articles dict
pol = ['L', 'LC', 'C', 'RC', 'R']
rep = ['VERY LOW', 'LOW', 'MIXED', 'HIGH', 'VERY HIGH']
flag = ['F', 'X', 'S']
labels = {}
import re

re_3986 = re.compile(r"^(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)(\?([^#]*))?(#(.*))?")
#Regular Expression to process web domains into chunks
wgo = re.compile("www.")
#For replacing www.

          
with open('bias.csv', 'r',encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        url = re_3986.match(row[4]).group(4)
        if url:
            name = wgo.sub("", url)
            if name in articles.keys():
                if name not in labels:
                    labels[name] = {'pol':0,'rep':0}
                if row[1] in pol:
                    labels[name]['pol'] = row[1]
                if row[2] in rep:
                    labels[name]['rep'] = row[2]
                
# Create the corpus
corpus = [articles[i] for i in articles.keys()]

# Clean text
clean_corpus = [clean(corpus[i]) for i in range(len(corpus))]
print("should equal docs", len(clean_corpus))


#Create TF-IDF valued matrix
vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(clean_corpus)

vocabulary = [] #list(vectorizer.vocabulary_.keys())
for i, feature in enumerate(vectorizer.get_feature_names()):
    if not any(char.isdigit() for char in feature):
        vocabulary.append(feature)

print("vocab size", len(vocabulary)) #perhaps further filter based on TF-IDF values, stemming, etc 

# Split into training and (holdout) testing sets
from sklearn.model_selection import train_test_split
test_label = 'pol'
article_ids = range(len(articles.keys()))
associated_labels = [labels[i][test_label] for i in articles.keys()]
X_train_ids, X_holdout_ids, y_train, y_holdout = train_test_split(article_ids, associated_labels, test_size=0.2, random_state=0)

# Grab domain feature vectors for each set to pass to classifier
#from operator import itemgetter
#X_train = itemgetter(*X_train_ids)(list(articles.keys()))
#X_holdout = itemgetter(*X_holdout_ids)(list(articles.keys()))
X_train = matrix[np.array(X_train_ids),:]
X_holdout = matrix[np.array(X_holdout_ids),:]

# Perform k-fold CV using the training set
k = 5; #number of folds
# select classification method
clf = RandomForestClassifier(oob_score=True,n_estimators=300)
# train classifier
clf.fit(X_train,y_train)
scores = cross_val_score(clf, X_train, y_train, cv=k)
print(scores)

# Test the classifier on the holdout set
print('Test Score:', clf.score(X_holdout,y_holdout))

## Create sample weights inversely proportional to class imbalances
#sample_counts = dict(Counter(y_train))
#training_set_length= len(y_train)
#keys = sorted(sample_counts.keys())
#sample_weights_per_class = {keys[k]: float(training_set_length)/sample_counts[k] for k in keys}
#sample_weights = []
#for i in range(training_set_length):
#    sample_weights.append(sample_weights_per_class[y_train[i]])
#            
#sample_weights = None
#        
#clf.fit(X_train, y_train,sample_weight=sample_weights)



