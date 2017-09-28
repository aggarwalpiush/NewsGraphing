# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 14:15:52 2017

@author: nfitch3
"""



#import matplotlib.pyplot as plt
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
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as SIA
#from sklearn import svm
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import ShuffleSplit, KFold
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import accuracy_score
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.feature_extraction.text import TfidfTransformer
#import scipy

from ipywidgets import IntProgress
from spacy.en import English


#import sys  
#from imp import reload
#reload(sys)  
#sys.setdefaultencoding('utf8')
#print(sys.getdefaultencoding())

def get_article_text():
    
    flag = 1
    
    nlp = spacy.load('en')
    #Load the Spacy English Language model


    client = MongoClient('mongodb://gdelt:meidnocEf1@11.7.124.26:27017/')
    #Connect to the GDELT Mongo database
    #Credentials might be different now, ask David
    db = client.gdelt.metadata


    re_3986 = re.compile(r"^(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)(\?([^#]*))?(#(.*))?")
    #Regular Expression to process web domains into chunks
    wgo = re.compile("www.")
    #For replacing www.
    whitelist = ["NOUN", "PROPN", "ADJ", "ADV"]
    #Types of words we'll look at


    #This opens up the MBFC labels which were scraped off their website
    bias = []
    biasnames = []
    pol = ['L', 'LC', 'C', 'RC', 'R'] #Political Bias
    rep = ['VERY LOW', 'LOW', 'MIXED', 'HIGH', 'VERY HIGH'] #Reporting Quality
    flag = ['F', 'X', 'S'] #Fake categories: Fake, Conspiracy, Satire
    cats = pol
    s2l = {}
    with open('bias.csv', 'r',encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            name = re_3986.match(row[4]).group(4)
            p = -1
            r = -1
            f = -1
            if row[1] in pol:
                p = pol.index(row[1])
                s2l[name] = row[1]
            if row[2] in rep:
                r = rep.index(row[2])
            if row[3] in flag:
                f = flag.index(row[3])
                s2l[name] = row[3]
            bias.append(row + [name, p, r, f, 1 if p == -1 else 0])
            
            if (p!=-1 or r!=-1) and name not in biasnames:
                biasnames.append(name)
            

    sample = 1000000
    stuff = db.find({},{'text':1,'sourceurl':1}).sort("_id",-1)#.limit(sample)

    arts = {}
    counts = {}
    print('Number of domains with at least one label', len(set(biasnames)))

    #Download articles and process them with SpaCy
    for obj in tqdm(stuff):
        if 'text' in obj:
            sdom = wgo.sub("", re_3986.match(obj['sourceurl']).group(4))
            if sdom in biasnames:
                if flag==1:
                    print('good')
                    #doc = nlp.tokenizer(obj['text'][:100*8])
                    #nlp.tagger(doc)
                    #Only break into tokens and give them part of speech tags
                    #doc_sentences = [sent.string.strip() for sent in doc.sents]
                    doc = ' '.join(obj['text'].split())
                    if sdom not in arts.keys():
                        arts[sdom] = []
                        counts[sdom] = 0
                    arts[sdom].append(doc)
                    counts[sdom] += 1
                else:
                    doc = nlp.tokenizer(obj['text'])
                    nlp.tagger(doc)
                    #Only break into tokens and give them part of speech tags
                    if sdom not in arts.keys():
                        arts[sdom] = []
                    arts[sdom].append(doc)
            
    print('Number of domains with text and at least one label', len(arts.keys()))
    print('Total number of articles: ', sum(counts.values()))

    return (arts)

#import pandas as pd
#df = pd.DataFrame(arts)
#df.to_csv('test.csv', index=False, header=False)
#
#import csv
#
#with open("output.csv", "wb") as f:
#    writer = csv.writer(f)
#    writer.writerows(a)
