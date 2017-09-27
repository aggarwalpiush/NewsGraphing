# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:16 2017

@author: nfitch3
"""


from __future__ import unicode_literals, print_function

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
import string
import numpy as np
import os
from sklearn import feature_extraction
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold
#from imblearn.under_sampling import RandomUnderSampler
#from imblearn.over_sampling import RandomOverSampler
#from sklearn.neural_network import MLPClassifier
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import naive_bayes

from dataset import get_article_text
import csv
#from collections import Counter
from sklearn.pipeline import Pipeline
#from nltk.stem import PorterStemmer,SnowballStemmer
import spacy
    
import sys
#from split_dataset import generate_hold_out_split

def remove_punctuation(myString):
    translator = str.maketrans('', '', string.punctuation)
    # Remove punctuation marks
    return myString.translate(translator)

def remove_stop_words(tokens):
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
    myTokens_stemmed = stem_words(' '.join(myTokens_no_stops))
    for token in myTokens_stemmed:
        if not any(char.isdigit() for char in token):
            myTokens_no_numbers.append(token)
    
    myString_cleaned = " ".join(myTokens_no_numbers)
    
    return myString_cleaned



pull_mongo_flag = True
 
# Read in text data
file_name = "gdelt_text.csv"
if pull_mongo_flag: #not (os.path.exists(file_name)):
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
            corpus.append(articles[sdom][0])
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
            
    # Create the corpus
    corpus = [articles[i] for i in articles.keys()]

print("Finished reading dataset")
print("corpus length:", len(corpus))

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
                    labels[name] = {'pol':'na','rep':'na'}
                if row[1] in pol:
                    labels[name]['pol'] = row[1]
                if row[2] in rep:
                    labels[name]['rep'] = row[2]
  
print("Finished reading labels", len(labels))

# Clean text
clean_corpus = [clean(corpus[i]) for i in range(len(corpus))]

# Reorganize labels
from sklearn.model_selection import train_test_split
test_label = 'pol' #'rep'
article_ids = []
associated_labels = []
s2i = {}
i2s = {}
for i,x in enumerate(articles.keys()):
    s2i[x] = i
    i2s[i] = x
    if labels[x][test_label] not in ['na']: # if non-empty
        if labels[x][test_label] in ['L','LC']:
            associated_labels.append(0)
        elif labels[x][test_label] in ['R','RC','C']: #center included here since R/RC is smaller category
            associated_labels.append(1)
        article_ids.append(i)

print("number of samples for testing/training: ", len(article_ids))
print("number of L/LC: ", (int(len(article_ids))-sum(associated_labels)))

# Split domains into training and (holdout) testing sets
X = article_ids
y = associated_labels
X_train_ids, X_holdout_ids, y_train, y_holdout = train_test_split(X, y, test_size=0.2, random_state=0)

print("training set length: ", len(y_train))
print("testing set length: ", len(y_holdout))

# Create pipeline of tasks
clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', RandomForestClassifier(oob_score=True,n_estimators=300))])

## Perform grid search to optimize parameters
#from sklearn.grid_search import GridSearchCV
#parameters = dict(feature_selection__k=[100, 200], 
#              random_forest__n_estimators=[50, 100, 200],
#              random_forest__min_samples_split=[2, 3, 4, 5, 10])
#
#clf = GridSearchCV(clf, parameters, n_jobs=-1)

#Create TF-IDF valued matrix on training set
X_train = [clean_corpus[i] for i in X_train_ids]
#vectorizer = TfidfVectorizer()
#matrix = vectorizer.fit_transform(X_train)

clf.fit(X_train,y_train)

#clf = RandomForestClassifier(oob_score=True,n_estimators=300)
#clf.fit(matrix,y_train)
#predictions = clf.predict(matrix)
print("training score: ", clf.score(X_train,y_train))

## Perform k-fold cv
#k = 5
#seed = 7
#cv = KFold(n_splits=k, random_state=seed)
#results = cross_val_score(clf,X,y,cv=cv)
#print("cv score: ", results.mean())

# Evaluate classifier on holdout set
X_holdout = [clean_corpus[i] for i in X_holdout_ids]
#matrix = vectorizer.transform(X_holdout)
#predictions = clf.predict(matrix)
predictions = clf.predict(X_holdout)
print("test score: ", np.mean(predictions==y_holdout))

# Write predictions to csv
with open('holdout.csv', "w",encoding='utf-8') as f:
    writer = csv.writer(f,lineterminator = '\n')
    for i,p in enumerate(predictions):
        writer.writerow([p,y_holdout[i]])
        


#vocabulary = [] #list(vectorizer.vocabulary_.keys())
#for i, feature in enumerate(vectorizer.get_feature_names()):
#    vocabulary.append(feature)
#
#print("training vocab size", len(vocabulary)) #perhaps further filter based on TF-IDF values, stemming, etc 


# Grab domain feature vectors for each set to pass to classifier
#from operator import itemgetter
#X_train = itemgetter(*X_train_ids)(list(articles.keys()))
#X_holdout = itemgetter(*X_holdout_ids)(list(articles.keys()))
#X_train = matrix_train[np.array(X_train_ids),:]
#X_holdout = matrix_holdout[np.array(X_holdout_ids),:]

# Perform k-fold CV using the training set
#k = 5; #number of folds
## select classification method
#clf = RandomForestClassifier(oob_score=True,n_estimators=300)
## train classifier
#clf.fit(X_train,y_train)
#scores = cross_val_score(clf, X_train, y_train, cv=k)
#print("cv scores", scores)

#Create TF-IDF valued matrix on holdout test set (except use IDF training values)
#X_holdout = vectorizer.transform([clean_corpus[i] for i in X_holdout_ids])
#
## Test the classifier on the holdout set
#print('Test Score:', clf.score(X_holdout,y_holdout))


# Read in predictions
y_holdout = []
predictions = []
with open('holdout.csv', 'r',encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        predictions.append(int(row[0]))
        y_holdout.append(int(row[1]))
            
            
# Print ROC curves + AUC score
from sklearn import metrics
fpr, tpr,_ = metrics.roc_curve(y_holdout,predictions)
AUC = metrics.auc(fpr,tpr)
print("AUC score: ", AUC)

## Plot ROC curve
#import matplotlib.pyplot as plt
#plt.plot(fpr,tpr,color='darkorange',lw=2,label='ROC Curve (area = %0.3f)' % AUC)
#plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Random Forest Receiver Operating Characteristic')
#plt.legend(loc="lower right")
#plt.savefig('roc.jpeg',bbox_inches='tight')

    
