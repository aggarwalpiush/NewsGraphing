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

from dataset import get_article_text

arts = get_article_text(flag=2)
# Read in dataset

vocab = set()
bivocab = set()

#Loop through all articles and create a big list of all occuring tokens
#We're doing tokens and bigrams
for (sdom, doc) in tqdm(arts):
for sdom in tqdm(arts.keys()):
    
    mycat = s2l[sdom]
    if mycat in cats:
        #for word in doc[:-1]:
        for word in arts[sdom]:
            if not word.is_stop and word.is_alpha and word.pos_ in whitelist:

                if not word.lemma_ in vocab:
                    vocab.add(word.lemma_)
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

for (sdom, doc) in tqdm(arts):
    
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
                
                site_raw_tc[sdom][v2i[word.lemma_]] += 1
                site_raw_ts[sdom][v2i[word.lemma_]] += c
                
                neigh = word.nbor()
                if not neigh.is_stop and neigh.pos_ in whitelist:
                    bigram = word.lemma_+" "+neigh.lemma_
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

clf = RandomForestClassifier(random_state=42, n_estimators=200)

X = np.concatenate((ttfidf.toarray(),site_tszv,site_tfdv,btfidf.toarray(),site_bszv,site_bfdv), axis=1)
print(X.shape)
y = np.array([cats.index(s2l[site]) for site in sites])
print(len(y))


cscore = cross_val_score(clf, X, y, cv=3)
print(cscore)
print(sum(cscore)/3)
clf.fit(X, y)
plt.plot(clf.feature_importances_)
plt.show()
mask = [i for i, x in enumerate(clf.feature_importances_) if x > 0.00035]
cscore = cross_val_score(clf, X[:, mask], y, cv=3)
print(cscore)
print(sum(cscore)/3)


cms = []
for train, test in KFold(n_splits=3).split(X):
    clf.fit(X[train,:][:,mask],y[train])
    cms.append(confusion_matrix(y[test], clf.predict(X[test,:][:,mask])))

print(sum(cms))
plt.imshow(sum(cms))
plt.show()
print(sum(sum(sum(cms))))



