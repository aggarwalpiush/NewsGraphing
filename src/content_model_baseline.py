# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:16 2017

@author: nfitch3
"""


from __future__ import unicode_literals, print_function

from sklearn.feature_extraction.text import TfidfVectorizer#, TfidfTransformer, CountVectorizer
#import string
import numpy as np
import os
#from sklearn import feature_extraction
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
#from sklearn.neural_network import MLPClassifier
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
#from sklearn import naive_bayes
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

from dataset import get_article_text
import csv
from collections import Counter
#from sklearn.pipeline import Pipeline
#from nltk.stem import PorterStemmer,SnowballStemmer
import sys
#from split_dataset import generate_hold_out_split
from sklearn.metrics import confusion_matrix
#from imblearn.under_sampling import RandomUnderSampler
#from imblearn.over_sampling import RandomOverSampler
#from nltk.stem.snowball import SnowballStemmer
#import nltk
#nltk.download('punkt')
from nltk import tokenize, word_tokenize

#import spacy
#import en_core_web_sm
#from gensim.models.doc2vec import Doc2Vec
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as SIA
from content_model_text_functions import *

BIASFILE = '../data/bias.csv'
CLF_ARGS = ['logreg','rf','svm']
LABEL_ARGS = ['bias','cred']
if len(sys.argv) < 2: # only script called
    # default classifier and task
    CLFNAME = 'logreg'
    LABELNAME = 'bias'
elif len(sys.argv) < 3: # only classifier given
    CLFNAME = sys.argv[1]
    # default task
    LABELNAME = 'bias'
else: # both classifier and task specified
    CLFNAME = sys.argv[1]
    LABELNAME = sys.argv[2]
    # must be a valid classifier
    if CLFNAME not in CLF_ARGS:
        sys.exit("Invalid classifier argument. Choose from " + str(CLF_ARGS) + " as first argument.")
    # must be a valid task
    if LABELNAME not in LABEL_ARGS:
        sys.exit("Invalid task argument. Choose from " + str(LABEL_ARGS) + " as second argument.")
        
print("Generating results for arguments: ",(LABELNAME, CLFNAME))

FILENAME = CLFNAME + '_' + LABELNAME
PATH = '../results/'
#os.makedirs(PATH)
 
# Read in text data
pull_mongo_flag = True
file_name = "../data/gdelt_text.csv"
cleanfile = "../data/gdelt_text_clean.csv"
clean_corpus = []
if not (os.path.exists(file_name)):
    
    #Get articles by domain name
    articles,corpus,s2l,i2s= get_article_text(BIASFILE)
    documents = articles.keys()
    
    #Write to file
    with open(file_name, "w",encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["article_id","sdom","text"])
        # Concatenate each domain's text into corpus
        #corpus = []
        for i,article in enumerate(corpus):
            # Write rows
            sdom = i2s[i]
            writer.writerow([i,sdom,article])
            
    f.close()
        
    with open(cleanfile, "w",encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["article_id","sdom","text"])
        # Concatenate each domain's text into corpus
        #corpus = []
        for i,article in enumerate(corpus):
            # Write rows
            sdom = i2s[i]
            # Clean corpus along the way
            clean_article = clean(article)
            writer.writerow([i,sdom,clean_article])
            
            # Create cleaned corpus
            clean_corpus.append(clean_article)
    
    f.close()

else:
    CLEAN = []
    # Read in articles from file_name
    maxInt = sys.maxsize
    decrement = True   
    articles = {}
    i2s = {}
    corpus = []
    with open(file_name, 'r',encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for i,row in enumerate(reader):
            while decrement: 
                # decrease the maxInt value by factor 10 
                # as long as the OverflowError occurs.
                decrement = False
                try:
                    csv.field_size_limit(maxInt)
                except OverflowError:
                    maxInt = int(maxInt/10)
                    decrement = True
            # each row = article ID, sdom, article text
            if row[1] not in articles:
                articles[row[1]] = []
            articles[row[1]].append(row[2])
            corpus.append(row[2])
            
            CLEAN.append(clean(row[2]))
            
            # Create the mapping from article ID to sdom name
            i2s[i] = row[1]
            
    articles_by_sdom = []
    with open(cleanfile, 'r',encoding='utf-8') as csvfile:
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
            # Read in the cleaned corpus
            clean_corpus.append(row[2])
            
            articles_by_sdom.append(row[1])
    
    sdom_counts = Counter(articles_by_sdom)
    with open('../data/sdom_by_article.csv', "w",encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["sdom","number of articles"])
        # Concatenate each domain's text into corpus
        #corpus = []
        for key in sdom_counts.keys():
            # Write rows
            writer.writerow([key,sdom_counts[key]])
    
            
print("Finished reading dataset")

# Read in MBFC labels from bias.csv
pol = ['L', 'LC', 'C', 'RC', 'R']
rep = ['VERY LOW', 'LOW', 'MIXED', 'HIGH', 'VERY HIGH']
flag = ['F', 'X', 'S']
labels = {}
import re

re_3986 = re.compile(r"^(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)(\?([^#]*))?(#(.*))?")
#Regular Expression to process web domains into chunks
wgo = re.compile("www.")
#For replacing www.

with open(BIASFILE, 'r',encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        url = re_3986.match(row[4]).group(4)
        if url:
            name = wgo.sub("", url)
            if name in articles.keys():
                if name not in labels:
                    labels[name] = {'bias':'na','cred':'na','flag':'na'}
                if row[1] in pol:
                    labels[name]['bias'] = row[1]
                rep_label = row[2]
                if rep_label not in rep:
                    rep_label = ' '.join(row[2].split()).upper()
                if rep_label in rep:
                    labels[name]['cred'] = rep_label
                if row[3] in flag:
                    labels[name]['flag'] = row[3]
  
print("Finished reading labels")


# Reorganize labels and choose classification problem
if LABELNAME == 'bias':
    l = 0
elif LABELNAME == 'cred':
    l = 1
label_type = ['bias','cred']
test_label_type = label_type[l]
test_labels = [[['L','LC'],['R','RC'],['na','C']],[['LOW','VERY LOW'],['HIGH', 'VERY HIGH'],['na','MIXED']]]
article_ids = []
associated_labels = []
associated_domains = []
i2l = {}
for i,x in enumerate(clean_corpus):
    sdom = i2s[i]
    if labels[sdom][test_label_type] not in test_labels[l][2]:
        if labels[sdom][test_label_type] in test_labels[l][0]:
            associated_labels.append(0)
            article_ids.append(i)
            i2l[i] = 0
            associated_domains.append(sdom)
        elif labels[sdom][test_label_type] in test_labels[l][1]:
            associated_labels.append(1)
            article_ids.append(i)
            i2l[i] = 1
            associated_domains.append(sdom)
#        elif labels[sdom][test_label] in ['C']:
#            associated_labels.append(2)
#            article_ids.append(i)
#            i2l[i] = 2
    elif l==1 and labels[sdom]['flag'] in flag:
        associated_labels.append(0)
        article_ids.append(i)
        i2l[i] = 0
        associated_domains.append(sdom)
        

print("number of articles with text : ", len(clean_corpus))    
print("number of samples for testing/training : ", len(article_ids))
print("distribution of labels : ", Counter(associated_labels))


# Split domains into training and (holdout) testing sets
from sklearn.model_selection import GroupKFold
X = article_ids
y = associated_labels
groups = associated_domains
group_kfold = GroupKFold(n_splits=5)

# split into training set (80%) and holdout set (20%) with non-overlapping groups
for train_ids,holdout_ids in group_kfold.split(X,y,groups):
    print("")

## then randomly give 40% more to the training set
## resulting in 80/20 split for training and holdout test sets
## first group by label
#holdout_ids_0 = [i for i in holdout_ids if y[i]==0]
#holdout_ids_1 = [i for i in holdout_ids if y[i]==1]
## randomly sample equal number from each group to take from holdout set and add to training set
#N = round(.4*len(holdout_ids))
#n = round(N/2)
#if test_label_type == 'pol': #bias
#    ids = list(np.random.choice(holdout_ids_0,n,replace=False)) + list(np.random.choice(holdout_ids_1,n,replace=False)) #bias
#else:
#    ids = holdout_ids_0 + list(np.random.choice(holdout_ids_1,abs(N-len(holdout_ids_0)),replace=False)) #cred
#new_train_ids = [i for i in holdout_ids if i not in ids]
#train_ids = list(train_ids) + new_train_ids 
#holdout_ids = ids

X_train_ids = [X[i] for i in train_ids]    
y_train = [y[i] for i in train_ids] 
groups_train = [groups[i] for i in train_ids]
X_holdout_ids = [X[i] for i in holdout_ids] 
y_holdout = [y[i] for i in holdout_ids] 
#X_train_ids, X_holdout_ids, y_train, y_holdout = train_test_split(X, y, test_size=0.2, stratify=y_sdom_group_ids, random_state=0)

print("training set length for CV: ", len(y_train))
print("holdout test set length: ", len(y_holdout))
print("holdout set label distribution: ", Counter(y_holdout))
print("training set label distribution: ", Counter(y_train))

# Write training and holdout test sets to ../data dir instead of PATH for future use
if LABELNAME == 'bias':
    files = ['training_set_bias.csv','holdout_set_bias.csv']
elif LABELNAME == 'cred':
    files = ['training_set_cred.csv','holdout_set_cred.csv']
    
split_ids = [train_ids] + [holdout_ids]
for i,file in enumerate(files):
    with open('../data/' + file, "w",encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        # Write header
        if LABELNAME == 'bias':
            writer.writerow(["article_id","sdom", "label_0_liberal_1_conservative"])
        elif LABELNAME == 'cred':
            writer.writerow(["article_id","sdom", "label_0_noncredible_1_credible"])

        for idx in split_ids[i]:
            # Write rows
            writer.writerow([article_ids[idx],associated_domains[idx],associated_labels[idx]])

## Create text embeddings using pretrained model to create vectors of dimension 300
#doc_model = Doc2Vec.load('doc2vec.bin')
#word_vectors = []
#for text in corpus: #clean_corpus:
#    word_vectors.append(text_to_vector(doc_model, text))
#
#print("word vectors created")

# Perform K-fold CV using training set
best_score = 0
k = 3
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
#skf = GroupKFold(n_splits=k)

# Create Oversampler and Undersampler objects
undersampler = RandomUnderSampler(random_state=42)
oversampler = RandomOverSampler(random_state=42)
    
for fold,(train_index, test_index) in enumerate(skf.split(X_train_ids, y_train, groups_train)):
    
    print("Fold: ", fold)
    
    # Separate training and test sets
    X_train_fold_ids = [X_train_ids[i] for i in train_index]
    X_test_fold_ids = [X_train_ids[i] for i in test_index]
    y_train_fold = [y_train[i] for i in train_index]
    y_test_fold = [y_train[i] for i in test_index]
    
    print("Train Label Distribution: ", Counter(y_train_fold))
    print("Test Label Distribution: ", Counter(y_test_fold))
        
    X_train = [clean_corpus[i] for i in X_train_fold_ids]
    X_test = [clean_corpus[i] for i in X_test_fold_ids]
    
#    # Undersample majority classes before training
#    X_train,y_train_fold = undersampler.fit_sample(X_train,y_train)
    
#    # Oversample minority classes before training
#    X_train,y_train_fold = oversampler.fit_sample(X_train,y_train_fold)

#    print("Train Label Distribution after over/under sampling: ", Counter(y_train_fold))

    # Create sample weights inversely proportional to class imbalances
    sample_counts = dict(Counter(y_train_fold))
    training_set_length= len(y_train_fold)
    keys = sorted(sample_counts.keys())
    sample_weights_per_class = {keys[k]: float(training_set_length)/sample_counts[k] for k in keys}
    sample_weights = []
    for i in range(training_set_length):
        sample_weights.append(sample_weights_per_class[y_train_fold[i]])
            
    #sample_weights = None
        

    # Create TF-IDF features
    tfidf = TfidfVectorizer(ngram_range=(1,2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
#    X_train_tfidf = [word_vectors[i] for i in X_train_fold_ids]
#    X_test_tfidf = [word_vectors[i] for i in X_test_fold_ids]

#    # Create pipeline of tasks
#    clf = Pipeline([
#            ('vect', CountVectorizer()),
#            ('tfidf', TfidfTransformer()),
#            ('clf', LogisticRegression())])
#            #('clf', RandomForestClassifier(oob_score=True,n_estimators=300))])
    
    # Instantiate classifier
    if CLFNAME == 'logreg':
        clf = LogisticRegression(penalty='l2',C=50)
    elif CLFNAME == 'svm':
        clf = SVC(C=1,probability=True)
    elif CLFNAME == 'rf':
        clf = RandomForestClassifier(random_state=42,oob_score=True,n_estimators=10)

    # Fit classifier
    print("fitting classifier...")
    clf.fit(X_train_tfidf, y_train_fold, sample_weight=sample_weights)
    
    # Make predictions
    print("making predictions...")
    predictions = clf.predict(X_test_tfidf)
    
    # Score it
    print("calculating acc score...")
    score = accuracy_score(y_test_fold, predictions)
    print('CV score: ', score)
    cms = confusion_matrix(y_test_fold, predictions)
    print(cms)
    
    # Keep best classifier
    if score > best_score:
        best_clf = clf
        best_tfidf = tfidf
        best_score = score
    
print('Best CV score: ', best_score)
   
        
## Perform grid search to optimize parameters
#from sklearn.grid_search import GridSearchCV
#parameters = dict(feature_selection__k=[100, 200], 
#              random_forest__n_estimators=[50, 100, 200],
#              random_forest__min_samples_split=[2, 3, 4, 5, 10])
#
#clf = GridSearchCV(clf, parameters, n_jobs=-1)

## Perform k-fold cv
#k = 5
#seed = 7
#cv = KFold(n_splits=k, random_state=seed)
#results = cross_val_score(clf,X,y)
#print("cv score: ", results.mean())
#

##Create TF-IDF valued matrix on training set
#X_train = [clean_corpus[i] for i in X_train_ids]
##vectorizer = TfidfVectorizer()
##matrix = vectorizer.fit_transform(X_train)
#
#clf.fit(X_train,y_train)
#
##clf = RandomForestClassifier(oob_score=True,n_estimators=300)
##clf.fit(matrix,y_train)
##predictions = clf.predict(matrix)
#print("training score: ", clf.score(X_train,y_train))



# Evaluate classifier on holdout set
X_holdout_ids = np.vstack(tuple([X[i] for i in holdout_ids]))
if test_label_type == 'rep':
    X_resampled,y_resampled = undersampler.fit_sample(X_holdout_ids,y_holdout)
else:
    X_resampled = X_holdout_ids
    y_resampled = y_holdout

X_holdout = [clean_corpus[i] for i in X_resampled.reshape(1, -1)[0]]
X_holdout_tfidf = best_tfidf.transform(X_holdout)

#X_holdout_tfidf = [word_vectors[i] for i in X_resampled.reshape(1, -1)[0]]

#predictions = clf.predict(matrix)
predictions = best_clf.predict(X_holdout_tfidf)
print("Holdout Label Distribution: ", Counter(y_resampled))
print("holdout test score: ", np.mean(predictions==y_resampled))

print (confusion_matrix(y_resampled, predictions))

predictions = best_clf.predict_proba(X_holdout_tfidf)[:,1]

# Records predictions and ROC/AUC
# Write predictions to csv
with open(PATH + FILENAME + '_results.csv', "w",encoding='utf-8') as f:
    writer = csv.writer(f,lineterminator = '\n')
    writer.writerow(['predictions','truth'])
    for i,p in enumerate(predictions):
        writer.writerow([p,y_holdout[i]])


# Get most informative features
if CLFNAME == 'logreg':
    coeffs = list(best_clf.coef_[0])
    n = 50
    best_feats_1 = sorted(range(len(coeffs)), key=lambda x: coeffs[x], reverse=True)[:n] # descending order
    best_feats_0 = sorted(range(len(coeffs)), key=lambda x: coeffs[x])[:n] # ascending order

    feature_names = best_tfidf.get_feature_names()
    top_n_words_0 = [(feature_names[i],coeffs[i]) for i in best_feats_0] #for class B where B < A
    top_n_words_1 = [(feature_names[i],coeffs[i]) for i in best_feats_1] #for class A where A > B
    
    top_n_words = [top_n_words_0] + [top_n_words_1]

    print("Vocabulary Length : ", len(best_tfidf.vocabulary_))
    if LABELNAME == 'bias':
        files = ['liberal_words.csv','conservative_words.csv']
    elif LABELNAME == 'cred':
        files = ['not_credible_words.csv','credible_words.csv']
    
    for i,category in enumerate(top_n_words):
        with open(PATH+files[i],'w',encoding='utf-8') as f:
            writer = csv.writer(f,lineterminator = '\n')
            writer.writerow(['word','coefficient value'])
            for words in category:
                writer.writerow(words)
                
    # Plot histogram of coefficients
    bins = np.linspace(np.min(coeffs), np.max(coeffs), 100)
    n,bins,patches = plt.hist(coeffs,bins,normed=True,alpha=.75)
    plt.xlabel('Coefficient Values')
    plt.ylabel('Probability')
    plt.title('Histogram of Log Reg Coefficients')
    plt.grid(True)
    plt.savefig(PATH+FILENAME+'_coeff_hist.png',bbox_inches='tight')
    plt.show()
    
# Get context
word2context = {}
word2sentiment = {}
dom2sentiment = {}
sentiment_stats_per_article = {}
#top_words = [w[0] for w in top_n_words_0] + [w[0] for w in top_n_words_1]
top_words = ['obamacare','affordable care act','donald trump','hillary clinton']#,'antifa','president trump','sputnik','crore','rs']
for w in top_words:
    sentiment_stats_per_article[w] = []
    word2context[w] = create_context(w,corpus,20,i2s)
    for line in word2context[w]:
        # reset with each article
        sent_scores = []
        if line['sentences']:
            sdom = line['sdom']
            label = labels[sdom]['bias']
            article_id = line['article_ID']
            w_tf = line['tf']
            # for each context sentence in article "line"
            for sentence in line['sentences']:
                sentiment = get_sentiment(w,sdom,sentence,labels)
                sent_scores.append(sentiment['score']) #collect sentiments for each sentence containing word in article: "line"
        
        # calculate sentiment stats
        article_avg = np.mean(sent_scores)
        article_min = np.min(sent_scores)
        article_max = np.max(sent_scores)
    
        # record stats
        sentiment_stats_per_article[w].append({'article_ID':article_id,'bias':labels[sdom]['bias'],'cred':labels[sdom]['cred'],'mean':article_avg,'min':article_min,'max':article_max})

#        tf = line['tf']
#        tf_lib = sum([len(line['sentences']) for line in word2context[w] if labels[line['sdom']]['bias'] in ['L','LC']])
#        tf_conserv = sum([len(line['sentences']) for line in word2context[w] if labels[line['sdom']]['bias'] in ['R','RC']])


plt.figure()
plt.hist([art['max'] for art in sentiment_stats_per_article[w] if art['bias'] in ["R","RC"]])
plt.figure()
plt.hist([art['max'] for art in sentiment_stats_per_article[w] if art['bias'] in ["L","LC"]])



## Plot source distribution histogram color-coded according to bias
#bar_data = []
#count=0
#i2name = {}
#names = sorted(sdom_counts.keys(), key=lambda x: sdom_counts[x], reverse=True)[:]
#for k in names:
#    if labels[k]['bias'] not in ['na']:
#        i2name[count] = k
#        if labels[k]['bias'] in ['L','LC']:
#            bar_data.append({'index': count,'height':sdom_counts[k],'color':'b'})
#            count+=1
#        elif labels[k]['bias'] in ['R','RC']:
#            bar_data.append({'index': count,'height':sdom_counts[k],'color':'r'})
#            count+=1
#        elif labels[k]['bias'] in ['C']:
#            bar_data.append({'index': count,'height':sdom_counts[k],'color':'green'})
#            count+=1
#        
#for data in bar_data:
#    plt.bar(data['index'], data['height'],align='center',color=data['color'])
##pos = [i for i in range(len(bar_data)) ]
##plt.xticks(pos, [data['index'] for data in bar_data])
#plt.xlabel('Source ID (color coded for bias)')
#plt.ylabel('Article Count per Source')
#plt.title('Distribution of Text Sources')
#plt.savefig('source_distribution_bias_color_coded.png',bbox_inches='tight')
#plt.show()
#
