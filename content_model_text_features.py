# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:04:51 2017

@author: nfitch3
"""


import matplotlib.pyplot as plt
import numpy as np
from pymongo import MongoClient
import tldextract
import math
import re
import pickle
from tqdm import tqdm_notebook as tqdm
import spacy
from numpy import dot
from numpy.linalg import norm
import csv
import random
import statistics
import copy
import itertools
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as SIA
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
import scipy

from sklearn.feature_extraction import text
from sklearn.linear_model import LogisticRegression

import string
from dataset import get_article_text

from spacy.en import English
from collections import Counter


def remove_punctuation(myString):
    translator = str.maketrans('', '', string.punctuation)
    # Remove punctuation marks
    return myString.translate(translator)

def remove_stop_words(tokens):
    from sklearn import feature_extraction
    return [w for w in tokens if w not in feature_extraction.text.ENGLISH_STOP_WORDS]

def stem_words(myString):
#    stemmer = SnowballStemmer('english') #PorterStemmer()
#    return [stemmer.stem(token) for token in tokens]
    stemmed = []
    nlp = spacy.load('en')
    for token in nlp(myString):
        if token.lemma_:
            token = token.lemma_
        stemmed.append(str(token))
    return stemmed

def return_top_k_keys(myDict,k):
    # Returns keys with largest k values (i.e. counts)
    return sorted(myDict, key=myDict.get, reverse=True)[:k]

def clean(myString):
    myTokens_no_numbers = []
    myTokens = remove_punctuation(myString.lower()).split()
    myTokens_no_stops = remove_stop_words(myTokens)
    #myTokens_stemmed = stem_words(' '.join(myTokens_no_stops))
    for token in myTokens_no_stops:
        if not any(char.isdigit() for char in token):
            myTokens_no_numbers.append(token)
    
    myString_cleaned = " ".join(myTokens_no_numbers)
    
    return myString_cleaned

def reality(label):
    if label in pol:
        cat = 'pol'
        if label in ['R','RC','C']:
            new_label = 0
        elif label in ['L','LC']:
            new_label = 1
    elif label in rep:
        cat = 'rep'
        if label in ['LOW','VERY LOW','MIXED']:
            new_label = 0
        elif label in ['HIGH','VERY HIGH']:
            new_label = 1
    
    return new_label


arts,s2l = get_article_text()
# Read in dataset


nlp = spacy.load('en')
#Load the Spacy English Language model

pol = ['L', 'LC', 'C', 'RC', 'R'] #Political Bias
rep = ['VERY LOW', 'LOW', 'MIXED', 'HIGH', 'VERY HIGH'] #Reporting Quality
flag = ['F', 'X', 'S'] #Fake categories: Fake, Conspiracy, Satire
cats = pol
whitelist = ["NOUN", "PROPN", "ADJ", "ADV"]

vocab = set()
bivocab = set()



for sdom in arts.keys():    
    #first clean text
    doc = arts[sdom][0]
    #doc = clean(doc)
    doc = nlp.make_doc(doc)
    nlp.tagger(doc)
    arts[sdom] = doc
# Create parts of speech tags

print("tagging finished")



#Loop through all articles and create a big list of all occuring tokens
#We're doing tokens and bigrams
#for (sdom, doc) in tqdm(arts):
for sdom in tqdm(arts.keys()):
    
    doc = arts[sdom]
    
    mycat = s2l[sdom]
    if mycat in cats:
        for word in doc[:-1]:
            if not word.is_stop and word.is_alpha and word.pos_ in whitelist:
                if not word.lemma_ in vocab:
                    if word.lemma_:
                        vocab.add(word.lemma_)
                    else:
                        vocab.add(word)
                neigh = word.nbor()
                if not neigh.is_stop and neigh.pos_ in whitelist:
                    bigram = word.lemma_+" "+neigh.lemma_
                    if not bigram in bivocab:
                        bivocab.add(bigram)
vsize = len(vocab)
print(vsize)
v2i = dict([(key, i) for i, key in enumerate(vocab)])
site_raw_tc = {}
site_raw_ts = {}

bvsize = len(bivocab)
print(bvsize)
bv2i = dict([(key, i) for i, key in enumerate(bivocab)])
site_raw_bc = {}
site_raw_bs = {}

#Build arrays for every site, containing counts of the terms and the average sentiment
#    Sentiment is collected for each term by adding the article's sentiment every
#time the term is detected, then dividing by the term count to get the mean

sa = SIA()

for sdom in tqdm(arts.keys()):
#for (sdom, doc) in tqdm(arts):
    
    doc = arts[sdom]
    
    mycat = s2l[sdom]
    if mycat in cats:
        if sdom not in site_raw_tc:
            site_raw_tc[sdom] = np.zeros(vsize)
            site_raw_ts[sdom] = np.zeros(vsize)
            site_raw_bc[sdom] = np.zeros(bvsize)
            site_raw_bs[sdom] = np.zeros(bvsize)
        c = sa.polarity_scores(doc.text)['compound']
        for word in doc[:-1]:
            if not word.is_stop and word.is_alpha and word.pos_ in whitelist:
                
                if word.lemma_:
                    site_raw_tc[sdom][v2i[word.lemma_]] += 1
                    site_raw_ts[sdom][v2i[word.lemma_]] += c
                else:
                    site_raw_tc[sdom][v2i[word]] += 1
                    site_raw_ts[sdom][v2i[word]] += c
                
                neigh = word.nbor()
                if not neigh.is_stop and neigh.pos_ in whitelist:
                    if neigh.lemma_:
                        if word.lemma_:
                            bigram = word.lemma_+" "+neigh.lemma_
                        else:
                            bigram = word+" "+neigh.lemma_
                    else:
                        if word.lemma_:
                            bigram = word.lemma_+" "+neigh
                        else:
                            bigram = word+" "+neigh
                    site_raw_bc[sdom][bv2i[bigram]] += 1
                    site_raw_bs[sdom][bv2i[bigram]] += c


sites = [k for k in site_raw_tc.keys()]
#List of sites
site_tcv = np.array([v for v in site_raw_tc.values()])
site_tsv = np.array([v for v in site_raw_ts.values()])
site_bcv = np.array([v for v in site_raw_bc.values()])
site_bsv = np.array([v for v in site_raw_bs.values()])
# Create 2D arrays for bigram and term counts and sentiments

site_tfv = site_tcv/np.sum(site_tcv, axis=1)[:,None]
site_tfv[np.isnan(site_tfv)] = 0
site_tsv = site_tsv/site_tcv
site_tsv[np.isnan(site_tsv)] = 0
site_bfv = site_bcv/np.sum(site_bcv, axis=1)[:,None]
site_bfv[np.isnan(site_bfv)] = 0
site_bsv = site_bsv/site_bcv
site_bsv[np.isnan(site_bsv)] = 0
#Calculate average sentiment and frequencies

s2c = dict([(site, s2l[site]) for site in sites])
cat_tcv = np.array([sum([site_raw_tc[site] for site in sites if s2l[site] == cat]) for cat in cats])
cat_tfv = cat_tcv/np.sum(cat_tcv, axis=1)[:, None]
cat_bcv = np.array([sum([site_raw_bc[site] for site in sites if s2l[site] == cat]) for cat in cats])
cat_bfv = cat_bcv/np.sum(cat_bcv, axis=1)[:, None]
#Calculate frequencies for each category

doc_tcv = np.sum(site_tcv, axis=0)
doc_tfv = doc_tcv/np.sum(doc_tcv)
doc_bcv = np.sum(site_bcv, axis=0)
doc_bfv = doc_bcv/np.sum(doc_bcv)
#Overall corpus frequencies

site_tszv = scipy.stats.mstats.zscore(site_tsv,axis=0)
site_tszv[np.isnan(site_tszv)] = 0
print("sent tz score" + str(site_tszv.shape))
#Z scores for term sentiment

site_bszv = scipy.stats.mstats.zscore(site_bsv,axis=0)
site_bszv[np.isnan(site_bszv)] = 0
print("sent bz score" + str(site_bszv.shape))
#Z scores for bigram sentiment

transformer = TfidfTransformer(smooth_idf=False)
ttfidf = transformer.fit_transform(site_tcv)
print("ttfidf" + str(ttfidf.shape))
btfidf = transformer.fit_transform(site_bcv)
print("btfidf" + str(btfidf.shape))
#Calculate TFIDF scores

site_tfdv = site_tfv - doc_tfv
site_bfdv = site_bfv - doc_bfv
#Difference in term frequency

#Run the models and score them

#clf = RandomForestClassifier(random_state=42, n_estimators=200)
clf = LogisticRegression()

X = np.concatenate((ttfidf.toarray(),site_tszv,site_tfdv,btfidf.toarray(),site_bszv,site_bfdv), axis=1)
print(X.shape)
#y = np.array([cats.index(s2l[site]) for site in sites])
y = np.array([reality(s2l[site]) for site in sites])
print(Counter(y))


cscore = cross_val_score(clf, X, y, cv=3)
print('CV scores: ', cscore)
print('avg: ', sum(cscore)/3)
clf.fit(X, y)
##plt.plot(clf.feature_importances_)
##plt.show()
#mask = [i for i, x in enumerate(clf.feature_importances_) if x > 0.00035]
mask = list(range(X.shape[1]))
#cscore = cross_val_score(clf, X[:, mask], y, cv=3)
#print('CV scores: ', cscore)
#print('avg: ', sum(cscore)/3)

from sklearn.cross_validation import train_test_split
X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.2, random_state=0)

cms = []
best_score = 0
for train, test in KFold(n_splits=3).split(X_train):
    clf.fit(X_train[train,:][:,mask],y_train[train])
    score = accuracy_score(y_train[test], clf.predict(X_train[test,:][:,mask]))
    print("cv score= ", score)
    if score > best_score:
        best_score = score
        best_clf = clf
        
    cms.append(confusion_matrix(y_train[test], clf.predict(X_train[test,:][:,mask])))

    print(cms)

#plt.imshow(sum(cms))
#plt.show()
print(sum(sum(sum(cms))))

# Write predictions to csv

predictions = best_clf.predict_proba(X_holdout)[:,1]
with open('holdout.csv', "w",encoding='utf-8') as f:
    writer = csv.writer(f,lineterminator = '\n')
    for i,p in enumerate(predictions):
        writer.writerow([p,y_holdout[i]])

# Read in predictions
y_holdout = []
predictions = []
with open('holdout.csv', 'r',encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        predictions.append(float(row[0]))
        y_holdout.append(float(row[1]))
            
            
# Print ROC curves + AUC score
from sklearn import metrics
fpr, tpr,_ = metrics.roc_curve(y_holdout,predictions)
AUC = metrics.auc(fpr,tpr)
print("AUC score: ", AUC)


# Plot ROC curve
plt.plot(fpr,tpr,color='darkorange',lw=2,label='ROC Curve (area = %0.3f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc.jpeg',bbox_inches='tight')




