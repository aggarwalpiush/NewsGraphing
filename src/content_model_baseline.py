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
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
#from sklearn.neural_network import MLPClassifier
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import naive_bayes
from sklearn.linear_model import LogisticRegression

from dataset import get_article_text
import csv
from collections import Counter
from sklearn.pipeline import Pipeline
#from nltk.stem import PorterStemmer,SnowballStemmer
import sys
#from split_dataset import generate_hold_out_split
from sklearn.metrics import confusion_matrix
#from imblearn.under_sampling import RandomUnderSampler
#from imblearn.over_sampling import RandomOverSampler
from nltk.stem.snowball import SnowballStemmer
#import nltk
#nltk.download('punkt')
from nltk import tokenize, word_tokenize

#import spacy
#import en_core_web_sm
from gensim.models.doc2vec import Doc2Vec
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as SIA

BIASFILE = "data/bias.csv"

def get_sentiment(w,sdom,contextString,labelDict):
    
    sa = SIA()

    bias_label = labelDict[sdom]['pol']
    cred_label = labelDict[sdom]['rep']
    sentiment = {'sdom':sdom,'bias':bias_label,'cred':cred_label,'score':sa.polarity_scores(contextString)['compound']}
        
#    score = [sentiment[i]["score"] for i in range(len(sentiment)) if sentiment[i]["bias"] in ["RC","R"] and sentiment[i]["cred"] in ["HIGH","VERY HIGH"]]
#    if score:
#        conservative_avg_score = np.mean(score)
#    else:
#        conservative_avg_score = 0
#        
#    score = [sentiment[i]["score"] for i in range(len(sentiment)) if sentiment[i]["bias"] in ["LC","L"] and sentiment[i]["cred"] in ["HIGH","VERY HIGH"]]
#    if score:
#        liberal_avg_score = np.mean(score)
#    else:
#        liberal_avg_score = 0
        
        
    return sentiment #liberal_avg_score, conservative_avg_score, sentiment


def text_to_vector(model, text):
    text_words = remove_stop_words(word_tokenize(text))
    model.random.seed(0)
    text_vector = model.infer_vector(text_words)
    return text_vector


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
#    nlp = spacy.load('en')
#    nlp = en_core_web_sm.load()
    for token in nlp(myString):
        if token.lemma_:
            token = token.lemma_
        stemmed.append(str(token))
    return stemmed

#def stem_words(myList):
#    new_list = []
#    stemmer = SnowballStemmer("english")
#    for w in myList:
#        new_list.append(stemmer.stem(w))
#    
#    return new_list

def return_top_k_keys(myDict,k):
    # Returns keys with largest k values (i.e. counts)
    return sorted(myDict, key=myDict.get, reverse=True)[:k]

#def clean(myString):
#    myTokens_no_numbers = []
#    myTokens = remove_punctuation(myString.lower()).split()
#    myTokens_no_stops = remove_stop_words(myTokens)
#    myTokens_stemmed = stem_words(' '.join(myTokens_no_stops))
#    for token in myTokens_stemmed: #myTokens_no_stops:
#        if not any(char.isdigit() for char in token):
#            myTokens_no_numbers.append(token)
#    
#    myString_cleaned = " ".join(myTokens_no_numbers)
#    
#    return myString_cleaned


def clean(myString):
    no_foreign_chars = myString.replace('“',' ').replace('”',' ').replace('’',"'")
    no_contractions = expand_contractions(no_foreign_chars.lower())
    no_punctuation = remove_punctuation(no_contractions)
    tokens = no_punctuation.split()
    no_stops = remove_stop_words(tokens)
#    stemmed_tokens = stem_words(' '.join(no_stops))
    no_numbers = []
#    for token in stemmed_tokens:
#        if not any(char.isdigit() for char in token):
#            no_numbers.append(token)
#    
#    myString_cleaned = " ".join(no_numbers)
    
    myString_cleaned = ' '.join(no_stops)

    return myString_cleaned
       
def expand_contractions(myString):
    english_contractions = { 
"ain't": "am not; are not; is not; has not; have not",
"aren't": "are not; am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"I'd": "I had / I would",
"I'd've": "I would have",
"I'll": "I shall / I will",
"I'll've": "I shall have / I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}
    new_s = list(myString.split())
    for i,w in enumerate(myString.split()):
        if w in english_contractions.keys():
            new_s[i] = english_contractions[w]
    return " ".join(new_s)


def create_context(contextWord,myCorpus,window,i2s):
    
    all_context = []
    for i,text in enumerate(myCorpus):
        context = {}
        if contextWord in text:
            # grab source its from
            context['sdom'] = i2s[i]
            context['article_ID'] = i
#            sents = tokenize.sent_tokenize(text)
            tokens = text.split()
#            indices = [i for i,x in enumerate(tokens) if x == contextWord]
            indices = [i for i, x in enumerate(tokens) if i<len(tokens)-1 and contextWord in tokens[i]+' '+tokens[i+1]]
#            context += [sent for sent in sents if contextWord in sent.lower().split()]
            context['sentences'] = []
            for i,index in enumerate(indices):
                if (index-window)>=0:
                    start = index-window
                else:
                    start = 0
                if (index+window)>len(tokens):
                    stop = len(tokens)
                    context['sentences'].append(tokens[start:stop])
                else:
                    stop = index+window
                    context['sentences'].append(tokens[start:stop+1])
        
            all_context.append(context)
    
    return all_context

 
# Read in text data
pull_mongo_flag = True
file_name = "data/gdelt_text.csv"
cleanfile = "data/gdelt_text_clean.csv"
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
    with open("gdelt_text_clean.csv", 'r',encoding='utf-8') as csvfile:
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
    with open('sdom_by_article.csv', "w",encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["sdom","number of articles"])
        # Concatenate each domain's text into corpus
        #corpus = []
        for key in sdom_counts.keys():
            # Write rows
            writer.writerow([key,sdom_counts[key]])
    
            
print("Finished reading dataset")

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

          
with open(BIASFILE, 'r',encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        url = re_3986.match(row[4]).group(4)
        if url:
            name = wgo.sub("", url)
            if name in articles.keys():
                if name not in labels:
                    labels[name] = {'pol':'na','rep':'na','flag':'na'}
                if row[1] in pol:
                    labels[name]['pol'] = row[1]
                rep_label = row[2]
                if rep_label not in rep:
                    rep_label = ' '.join(row[2].split()).upper()
                if rep_label in rep:
                    labels[name]['rep'] = rep_label
                if row[3] in flag:
                    labels[name]['flag'] = row[3]
  
print("Finished reading labels")


# Reorganize labels and choose classification problem
l = 0 # 0 = bias and 1 = credibility
label_type = ['pol','rep']
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
    print(len(train_ids),len(holdout_ids))

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

# Write training and holdout test sets
with open('training_set_bias.csv', "w",encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["article_id","sdom","bias_label_0_liberal_1_conservative"])
        # Concatenate each domain's text into corpus
        #corpus = []
        for idx in train_ids:
            # Write rows
            writer.writerow([article_ids[idx],associated_domains[idx],associated_labels[idx]])
            
with open('holdout_set_bias.csv', "w",encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["article_id","sdom","bias_label_0_liberal_1_conservative"])
        # Concatenate each domain's text into corpus
        #corpus = []
        for idx in holdout_ids:
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
    
    # Fit the classifier using pipeline
#    clf = SVC(C=1,probability=True)
    clf = LogisticRegression(penalty='l2',C=50)
#    clf = RandomForestClassifier(random_state=42,oob_score=True,n_estimators=10)
    print("fitting classifier...")
    clf.fit(X_train_tfidf, y_train_fold, sample_weight=sample_weights)
    print("making predictions...")
    predictions = clf.predict(X_test_tfidf)
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


# Get most informative features
coeffs = list(best_clf.coef_[0])
n = 50
best_feats_1 = sorted(range(len(coeffs)), key=lambda x: coeffs[x], reverse=True)[:n] # descending order
best_feats_0 = sorted(range(len(coeffs)), key=lambda x: coeffs[x])[:n] # ascending order

feature_names = best_tfidf.get_feature_names()
top_n_words_0 = [(feature_names[i],coeffs[i]) for i in best_feats_0] #for class B where B < A
top_n_words_1 = [(feature_names[i],coeffs[i]) for i in best_feats_1] #for class A where A > B

path = 'results/content_baseline/context/'
os.makedirs(path)

print("Vocabulary Length : ", len(best_tfidf.vocabulary_))
print("Most informative label 0 words : ")
with open(path+'liberal_words.csv','w',encoding='utf-8') as f:
    writer = csv.writer(f,lineterminator = '\n')
    writer.writerow(['word','coefficient value'])
    for i in range(len(top_n_words_0)):
        print(top_n_words_0[i]) 
        writer.writerow(top_n_words_0[i])
print("Most informative label 1 words : ")
with open(path+'conservative_words.csv','w',encoding='utf-8') as f:
    writer = csv.writer(f,lineterminator = '\n')
    writer.writerow(['word','coefficient value'])
    for i in range(len(top_n_words_1)):
        print(top_n_words_1[i]) 
        writer.writerow(top_n_words_1[i])

    
# Get context
word2context ={}
word2sentiment = {}
sent_scores = []
dom2sentiment = {}
avg_sent_score_per_article = {}
#top_words = [w[0] for w in top_n_words_0] + [w[0] for w in top_n_words_1]
top_words = ['obamacare']#,'antifa','president trump','sputnik','crore','rs']
for w in top_words:
    word2context[w] = create_context(w,clean_corpus,20,i2s)
    for line in word2context[w]:
        if line['sentences']:
            sdom = line['sdom']
            # for each context sentence in article "line"
            for sentence in line['sentences']:
                context_sentence = ' '.join(sentence) # create string
                sentiment = get_sentiment(w,sdom,context_sentence,labels)
                sent_scores.append(sentiment['score']) #collect sentiments for each sentence containing word in article: "line"
                
    avg_sent_score_per_article[w] = np.mean(sent_scores)
#            if sdom in dom2sentiment.keys():
#                dom2sentiment[sdom]['liberal'] = dom2sentiment[sdom]['liberal'] + avg_lib
#                dom2sentiment[sdom]['conservative'] = dom2sentiment[sdom]['conservative'] + avg_conserv
#            else:
#                dom2sentiment[sdom] = {'liberal':avg_lib,'conservative':avg_conserv}
    
    #word2sentiment[w] = {'liberal':avg_lib,'conservative':avg_conserv} # average sentiment per word for across all articles in corpus



# Generate topics
#from gensim import corpora, models
#topic_dict = corpora.Dictionary(word2context[w])
#topic_corpus = [topic_dict.doc2bow(text) for text in word2context[w]]
#LDA_model = gensim.models.ldamodel.Ldamodel(topic_corpus,num_topics=2,id2word=topic_dict,passes=20)


# Write predictions to csv
with open('results.csv', "w",encoding='utf-8') as f:
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

data = []
AUC = []
# Read in predictions
y_holdout = []
predictions = []
coeffs = []
files = ['logreg_bias_results.csv','logreg_bias_paragraph_vectors_results.csv','randforest_bias_results.csv']
roc_labels = ['Log Reg (TF-IDF) (area = %0.3f)','Log Reg (Paragraph Vectors) (area = %0.3f)','Random Forest (TF-IDF) (area = %0.3f)']
for f in files:
    with open(f, 'r',encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            predictions.append(float(row[0]))
            y_holdout.append(float(row[1]))
#           coeffs.append(float(row[2]))
            
            
    # Print ROC curves + AUC score
    from sklearn import metrics
    fpr, tpr, roc_thresholds = metrics.roc_curve(y_holdout,predictions)
    auc = metrics.auc(fpr,tpr)
    print("AUC score: ", auc)

    data.append([fpr,tpr])
    AUC.append(auc)
    
    # Print accuracy/cms
    print( "cms for file : ",f)
    print (np.mean(y_resampled==predictions))


# Plot ROC curve
data = [[fpr,tpr]]
AUC = [auc]
roc_labels=['Log Reg (TF-IDF) (area = %0.3f)']
import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
for i,d in enumerate(data):
    plt.plot(data[i][0],data[i][1],lw=2,label=roc_labels[i] % AUC[i])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc.jpeg',bbox_inches='tight')
plt.show()


plt.hist([x for i,x in enumerate(predictions) if y_resampled[i]==0])
plt.hist([x for i,x in enumerate(predictions) if y_resampled[i]==1])
#plt.hist(y_resampled)
plt.show()
sum(predictions < .5)/float(len(predictions))


# Plot histogram of coefficients
bins = np.linspace(np.min(coeffs), np.max(coeffs), 50)
n,bins,patches = plt.hist(coeffs,bins,normed=True,alpha=.75)
plt.xlabel('Coefficient Values')
plt.ylabel('Probability')
plt.title('Histogram of Log Reg Coefficients')
plt.grid(True)
plt.savefig('coeff_hist.jpeg',bbox_inches='tight')
plt.show()

# Plot source distribution histogram color-coded according to bias
bar_data = []
count=0
i2name = {}
names = sorted(sdom_counts.keys(), key=lambda x: sdom_counts[x], reverse=True)[:]
for k in names:
    if labels[k]['pol'] not in ['na']:
        i2name[count] = k
        if labels[k]['pol'] in ['L','LC']:
            bar_data.append({'index': count,'height':sdom_counts[k],'color':'b'})
            count+=1
        elif labels[k]['pol'] in ['R','RC']:
            bar_data.append({'index': count,'height':sdom_counts[k],'color':'r'})
            count+=1
        elif labels[k]['pol'] in ['C']:
            bar_data.append({'index': count,'height':sdom_counts[k],'color':'green'})
            count+=1
        
for data in bar_data:
    plt.bar(data['index'], data['height'],align='center',color=data['color'])
#pos = [i for i in range(len(bar_data)) ]
#plt.xticks(pos, [data['index'] for data in bar_data])
plt.xlabel('Source ID (color coded for bias)')
plt.ylabel('Article Count per Source')
plt.title('Distribution of Text Sources')
plt.savefig('source_distribution_bias_color_coded.jpeg',bbox_inches='tight')
plt.show()

for i,x in enumerate(zip(n,bins)):
    print(x)

print(len(coeffs))
