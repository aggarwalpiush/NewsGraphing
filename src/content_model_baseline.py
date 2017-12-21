
from __future__ import unicode_literals, print_function

import logging
import re
from sklearn.feature_extraction.text import TfidfVectorizer#, TfidfTransformer, CountVectorizer
from sklearn.model_selection import GroupKFold
#import string
import numpy as np
import os
import dataset
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
from dataset import get_article_text#, writetextcsv, writecleancsv
import csv
from collections import Counter
#from sklearn.pipeline import Pipeline
#from nltk.stem import PorterStemmer,SnowballStemmer
import sys
#from split_dataset import generate_hold_out_split
from sklearn.metrics import confusion_matrix
#from nltk.stem.snowball import SnowballStemmer
#import nltk
#nltk.download('punkt')
from nltk import tokenize, word_tokenize
#import spacy
#import en_core_web_sm
#from gensim.models.doc2vec import Doc2Vec
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as SIA
from content_model_text_functions import * #clean, create_context

def record_results(data,FILENAME,name_arg=None):
    predictions = data[0]
    truth = data[1]
    
    if name_arg is None:
        NAME = FILENAME
    else:
        NAME = FILENAME +'_'+str(name_arg)
        
    with open(PATH + NAME + '_results.csv', "w",encoding='utf-8') as f:
        writer = csv.writer(f,lineterminator = '\n')
        writer.writerow(['probability_label_1','truth'])
        for i,p in enumerate(predictions):
            writer.writerow([p,truth[i]])
            
    return 

def evaluate_classifier(predictions,truth):
    
    # Check that predictions are integers
    if not all(type(item)==int for item in predictions):
        predictions = [1 if p >=.5 else 0 for p in predictions]
        
    # Compute Accuracy
    score = accuracy_score(truth, predictions)
    
    # Print confusion matrix
    cms = confusion_matrix(truth, predictions)
    print(cms)
    
    return score, cms

def fit_and_predict(training_set,test_set,CLFNAME,sample_weights=None):
    X_train = training_set[0]
    y_train = training_set[1]
        
    X_test = test_set[0]
    y_test = test_set[1]
        
    # Instantiate classifier
    if CLFNAME == 'logreg':
        clf = LogisticRegression(penalty='l2',C=50)
    elif CLFNAME == 'svm':
        clf = SVC(C=1,probability=True)
    elif CLFNAME == 'rf':
        clf = RandomForestClassifier(random_state=42,oob_score=True,n_estimators=10)

    # Fit classifier
    print("fitting classifier...")
    clf.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Make predictions
    print("making predictions...")
    predictions = clf.predict_proba(X_test)[:,1]
        
    return clf, predictions

def k_fold_CV(k,X,y,CLFNAME):
    
    # Perform K-fold CV using training set
    best_score = 0
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    #skf = GroupKFold(n_splits=k)

    # Create Oversampler and Undersampler objects
    undersampler = RandomUnderSampler(random_state=42)
    oversampler = RandomOverSampler(random_state=42)

    
    for fold,(train_index, test_index) in enumerate(skf.split(X,y)):
    
        print("Fold: ", fold)
    
        # Separate training and test sets
        X_train_fold = [X[i] for i in train_index]
        X_test_fold = [X[i] for i in test_index]
        y_train_fold = [y[i] for i in train_index]
        y_test_fold = [y[i] for i in test_index]
    
        print("Train Label Distribution: ", Counter(y_train_fold))
        print("Test Label Distribution: ", Counter(y_test_fold))

        # Create sample weights inversely proportional to class imbalances
        sample_counts = dict(Counter(y_train_fold))
        training_set_length= len(y_train_fold)
        keys = sorted(sample_counts.keys())
        sample_weights_per_class = {keys[k]: float(training_set_length)/sample_counts[k] for k in keys}
        sample_weights = []
        for i in range(training_set_length):
            sample_weights.append(sample_weights_per_class[y_train_fold[i]])
            
  
        # Create TF-IDF features
        tfidf = TfidfVectorizer(ngram_range=(1,2))
        X_train_tfidf = tfidf.fit_transform(X_train_fold)
        X_test_tfidf = tfidf.transform(X_test_fold)
        
#        X_train_tfidf = [word_vectors[i] for i in X_train_fold_ids]
#        X_test_tfidf = [word_vectors[i] for i in X_test_fold_ids]
    
#        # Create pipeline of tasks
#        clf = Pipeline([
#                ('vect', CountVectorizer()),
#                ('tfidf', TfidfTransformer()),
#                ('clf', LogisticRegression())])
#                #('clf', RandomForestClassifier(oob_score=True,n_estimators=300))])
        
        # Fit classifier and get predictions
        training_set = [X_train_tfidf,y_train_fold]
        test_set = [X_test_tfidf,y_test_fold]
        clf, predictions = fit_and_predict(training_set,test_set,CLFNAME,sample_weights)
        
        # Evaluate classifier
        score,cms = evaluate_classifier(predictions,y_test_fold)
    
        print('Fold score: ', score)
    
        # Keep best classifier
        if score > best_score:
            best_clf = clf
            best_tfidf = tfidf
            best_score = score
    
    print('Best CV score: ', best_score)

    return best_clf, best_tfidf   
  


def get_problem_set(dataset,LABELNAME,labels,i2s):
    # Aggregate labels and associated article/domain information by chosen classification task
    test_labels = {'bias':[['L','LC'],['R','RC'],['na','C']],'cred':[['LOW','VERY LOW'],['HIGH', 'VERY HIGH'],['na','MIXED']]}
    article_ids = []
    associated_labels = []
    associated_domains = []
    i2l = {}
    for i,x in enumerate(dataset):
        sdom = i2s[i]
        if labels[sdom][LABELNAME] not in test_labels[LABELNAME][2]:
            if labels[sdom][LABELNAME] in test_labels[LABELNAME][0]:
                associated_labels.append(0)
                article_ids.append(i)
                i2l[i] = 0
                associated_domains.append(sdom)
            elif labels[sdom][LABELNAME] in test_labels[LABELNAME][1]:
                associated_labels.append(1)
                article_ids.append(i)
                i2l[i] = 1
                associated_domains.append(sdom)
        elif LABELNAME == 'cred' and labels[sdom]['flag'] in flag:
            associated_labels.append(0)
            article_ids.append(i)
            i2l[i] = 0
            associated_domains.append(sdom)
        
    return article_ids, associated_labels, associated_domains, i2l


def generate_sentiment_features(top_words,labels,X_train_ids,X_holdout_ids,corpus,i2s):
    
    print('generating sentiment features...')
    # Get context
    word2context = {}
    sentiment_stats_per_article = {}
    training_feats = []
    holdout_feats = []

    # for each word in top_words, grab sentences that contain the word
    for w_idx,w in enumerate(top_words):
        training_feats.append([0]*len(X_train_ids))
        holdout_feats.append([0]*len(X_holdout_ids))
        sentiment_stats_per_article[w] = []
        word2context[w] = create_context(w,corpus,20,i2s)
        
        # loop through each article
        for line in word2context[w]:
            sent_scores = [] # reset for each new article
            sdom = line['sdom']
            label = labels[sdom]['bias']
            article_id = line['article_ID']
        
            if line['sentences']: # and label not in ['C']:
                # for each context sentence in article "line"
                for sentence in line['sentences']:
                    sentiment = get_sentiment(w,sdom,sentence,labels)
                    sent_scores.append(sentiment['score']) #collect sentiments for each sentence containing word in article: "line"
        
                # calculate sentiment stats for current article
                article_avg = np.mean(sent_scores)
                article_min = np.min(sent_scores)
                article_max = np.max(sent_scores)
    
                # record stats per article in 'line' per word 'w'
                sentiment_stats_per_article[w].append({'article_ID':article_id,'bias':labels[sdom]['bias'],'cred':labels[sdom]['cred'],'mean':article_avg,'min':article_min,'max':article_max})
 
                # check that article is in labeled training dataset
                if article_id in X_train_ids:
                    idx = X_train_ids.index(article_id)
                    training_feats[w_idx][idx] = article_avg

                # check that article is in labeled holdout dataset
                if article_id in X_holdout_ids:
                    idx = X_holdout_ids.index(article_id)
                    holdout_feats[w_idx][idx] = article_avg
        
        
    # transform list of lists into numpy arrays so that each column is feature for word 'w'
    training_vector = np.transpose(np.array(training_feats))
    holdout_vector = np.transpose(np.array(holdout_feats))
    
    return training_vector, holdout_vector, sentiment_stats_per_article, word2context


def generate_sentiment_features_ALT(top_words,labels,X_train_ids,X_holdout_ids,corpus,i2s):
    
    sa = SIA()
    
    print('generating sentiment features...')
    # Get context
    word2context = {}
    sentiment_stats_per_article = {}
    training_feats = []
    holdout_feats = []

    # for each word in top_words, grab sentences that contain the word
    for w_idx,w in enumerate(top_words):
        training_feats.append([0]*len(X_train_ids))
        holdout_feats.append([0]*len(X_holdout_ids))
        sentiment_stats_per_article[w] = []
        word2context[w] = create_context(w,corpus,20,i2s)
        
        # loop through each article
        first_pass = True
        first=True
        for line in word2context[w]:
            sdom = line['sdom']
            label = labels[sdom]['bias']
            article_id = line['article_ID']
        
            # concatenate all text by label
            if labels[sdom]['bias'] in ["R"]:
                temp = ' '.join(line['sentences'])
                if first_pass:
                   conservative_terms = [temp]
                   first_pass = False
                else:
                   conservative_terms = [' '.join([conservative_terms[0],temp])]
    
            if labels[sdom]['bias'] in ["L"]:
                temp = ' '.join(line['sentences'])
                if first:
                    liberal_terms = [temp]
                    first = False
                else:
                    liberal_terms = [' '.join([liberal_terms[0],temp])]


        # remove shared terms from sentences and then compute sentiment score per article
        shared_terms = list(set(liberal_terms[0].split()) & set(conservative_terms[0].split()))
        for art in word2context[w]:
            article_id = art['article_ID']
            temp = ' '.join(art['sentences'])
            art_leftovers = ' '.join([word for word in temp.split() if word not in shared_terms])
            score = sa.polarity_scores(art_leftovers)['compound']
            sentiment_stats_per_article[w].append({'article_id':article_id,'bias':labels[sdom]['bias'],'cred':labels[sdom]['cred'],'sentiment_score':score})

            # check that article is in labeled training dataset
            if article_id in X_train_ids:
                idx = X_train_ids.index(article_id)
                training_feats[w_idx][idx] = score

            # check that article is in labeled holdout dataset
            if article_id in X_holdout_ids:
                idx = X_holdout_ids.index(article_id)
                holdout_feats[w_idx][idx] = score
        
#        # replace articles without context word with label average
#        avg_lib_score =np.mean([line['sentiment_score'] for line in sentiment_stats_per_article[w] if line['bias'] in ["L","LC"]])
#        avg_conserv_score = np.mean([line['sentiment_score'] for line in sentiment_stats_per_article[w] if line['bias'] in ["R","RC"]])
#        
#        zero_idx = np.where(training_feats==0)[0]
#        lib_idx = np.where()
#        training_feats = np.array(training_feats)
#        training_feats[zero_idx]
#        for idx in zero_idx:
#            if 
#            training_feats[idx] = avg_lib_score
        
    # transform list of lists into numpy arrays so that each column is feature for word 'w'
    training_vector = np.transpose(np.array(training_feats))
    holdout_vector = np.transpose(np.array(holdout_feats))
    
    return training_vector, holdout_vector, sentiment_stats_per_article, word2context

# main
logging.basicConfig(format='%(levelname)s %(asctime)-15s %(message)s', level=logging.INFO) 

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
      
print("Generating results for arguments: ", (LABELNAME, CLFNAME)) 

FILENAME = CLFNAME + '_' + LABELNAME
PATH = '../results/'
#os.makedirs(PATH)
 
# Read in text data
pull_mongo_flag = True
file_name = "../data/gdelt_text.csv"
cleanfile = "../data/gdelt_text_clean.csv"
clean_corpus = None
if not (os.path.exists(file_name)):
    
    logging.info("Downloading data from database") 
    #Get articles by domain name
    articles,corpus,s2l,i2s= get_article_text(BIASFILE)
    documents = articles.keys()
    logging.info("writing out text corpus")
    writetextcsv(file_name, corpus, i2s)
    logging.info("writing out cleaned up text corpus")
    clean_corpus = writecleancsv(clean, cleanfile, corpus, i2s)

else:
    logging.info("Reading data from cache")
    CLEAN = []
    # Read in articles from file_name
    maxInt = sys.maxsize
    decrement = True   
    articles = {}
    i2s = {}
    corpus = []
    clean_corpus = []
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
                    logging.info('decrementing csv field size limit.')
            # each row = article ID, sdom, article text
            if row[1] not in articles:
                articles[row[1]] = []
            articles[row[1]].append(row[2])
            corpus.append(row[2])
            
            CLEAN.append(clean(row[2]))
            
            # Create the mapping from article ID to sdom name
            i2s[i] = row[1]
            
    logging.info("Reading data from clean file cache")    
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
    datapath = '../data/sdom_by_article.csv'
    logging.info("Writing count data to path {}".format(datapath))  
    with open(datapath, "w",encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["sdom","number of articles"])
        # Concatenate each domain's text into corpus
        #corpus = []
        for key in sdom_counts.keys():
            # Write rows
            writer.writerow([key,sdom_counts[key]])
    
            
logging.info("Finished reading dataset")

# Read in MBFC labels from bias.csv
pol = ['L', 'LC', 'C', 'RC', 'R']
rep = ['VERY LOW', 'LOW', 'MIXED', 'HIGH', 'VERY HIGH']
flag = ['F', 'X', 'S']
labels = {}

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
  
logging.info("Finished reading labels")

# aggregate total labeled dataset per given problem arguments
logging.info('Aggregate labeled dataset for chosen classification task') 
ARTICLE_IDs, ARTICLE_LABELS, ARTICLE_DOMAINS, article2label = get_problem_set(clean_corpus,LABELNAME,labels,i2s)

print("number of articles with text : ", len(clean_corpus))    
print("number of samples for testing/training : ", len(ARTICLE_IDs))
print("distribution of labels : ", Counter(ARTICLE_LABELS))


# Split labeled dataset into training set (80%) and holdout set (20%) with non-overlapping groups = ARTICLE_DOMAINS
group_kfold = GroupKFold(n_splits=5)
splits = list(group_kfold.split(ARTICLE_IDs,ARTICLE_LABELS,ARTICLE_DOMAINS))

# splits WITH overlapping domains
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
splits = list(skf.split(ARTICLE_IDs,ARTICLE_LABELS))

# select train/test indices
train_indicies,holdout_indicies = splits[-1]

# create training set
X_train_ids = [ARTICLE_IDs[i] for i in train_indicies]   
X_TRAIN = [clean_corpus[i] for i in X_train_ids] 
y_TRAIN = [ARTICLE_LABELS[i] for i in train_indicies] 

# create holdout test set
X_holdout_ids = [ARTICLE_IDs[i] for i in holdout_indicies]
X_HOLDOUT = [clean_corpus[i] for i in X_holdout_ids] 
y_HOLDOUT = [ARTICLE_LABELS[i] for i in holdout_indicies] 

print("training set length for CV: ", len(y_TRAIN))
print("holdout test set length: ", len(y_HOLDOUT))
print("holdout set label distribution: ", Counter(y_HOLDOUT))
print("training set label distribution: ", Counter(y_TRAIN))

# Write training and holdout test sets to ../data dir instead of PATH for future use
if LABELNAME == 'bias':
    files = ['training_set_bias.csv','holdout_set_bias.csv']
elif LABELNAME == 'cred':
    files = ['training_set_cred.csv','holdout_set_cred.csv']
    
split_ids = [train_indicies] + [holdout_indicies]
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
            writer.writerow([ARTICLE_IDs[idx],ARTICLE_DOMAINS[idx],ARTICLE_LABELS[idx]])

## Create text embeddings using pretrained model to create vectors of dimension 300
#doc_model = Doc2Vec.load('doc2vec.bin')
#word_vectors = []
#for text in corpus: #clean_corpus:
#    word_vectors.append(text_to_vector(doc_model, text))
#
#print("word vectors created")

# Perform k-fold cv
k = 3
data = clean_corpus
best_clf, best_tfidf = k_fold_CV(k,X_TRAIN,y_TRAIN,CLFNAME)


# Evaluate classifier on holdout set
# Create Oversampler and Undersampler objects
undersampler = RandomUnderSampler(random_state=42)
oversampler = RandomOverSampler(random_state=42)
X_holdout_ids = np.vstack(tuple([ARTICLE_IDs[i] for i in holdout_indicies]))
if LABELNAME == 'cred':
    X_resampled,y_resampled = undersampler.fit_sample(X_holdout_ids,y_HOLDOUT)
else:
    X_resampled = X_holdout_ids
    y_resampled = y_HOLDOUT

X_holdout = [clean_corpus[i] for i in X_resampled.reshape(1, -1)[0]]
X_holdout_tfidf = best_tfidf.transform(X_holdout)

#X_holdout_tfidf = [word_vectors[i] for i in X_resampled.reshape(1, -1)[0]]

#predictions = clf.predict(matrix)
predictions = best_clf.predict(X_holdout_tfidf)
print("Holdout Label Distribution: ", Counter(y_resampled))
print("holdout test score: ", np.mean(predictions==y_resampled))

print (confusion_matrix(y_resampled, predictions))

predictions = best_clf.predict_proba(X_holdout_tfidf)[:,1]
# Write predictions to csv
record_results([predictions,y_HOLDOUT],FILENAME)


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
 

# create TF-IDF matrix on entire training set
X_train_tfidf = best_tfidf.fit_transform(X_TRAIN)
# create TF-IDF matrix on holdout test set
X_holdout_tfidf = best_tfidf.transform(X_HOLDOUT)

X_holdout_ids = [ARTICLE_IDs[i] for i in holdout_indicies]
# generate sentiment feature vectors
#top_words = [w[0] for w in top_n_words_0] + [w[0] for w in top_n_words_1]
top_words = ['climate change','crore','obamacare','affordable care act']#,'donald trump','hillary clinton','climate change']#,'antifa','president trump','sputnik','crore','rs']
training_sentiment_vector, holdout_sentiment_vector, sentiment_stats_per_article, word2context = generate_sentiment_features_ALT(top_words,labels,X_train_ids,X_holdout_ids,corpus,i2s)


#concatenate TF-IDF and sentiment features
from scipy import sparse
X_train_combined = sparse.hstack([X_train_tfidf,training_sentiment_vector])
X_holdout_combined = sparse.hstack([X_holdout_tfidf,holdout_sentiment_vector])
# train CLFNAME classifier
best_clf.fit(X_train_combined,y_TRAIN)

# make predictions
predictions = best_clf.predict_proba(X_holdout_combined)[:,1]
#print(best_clf.classes_)
# evaluate classifier performance
acc,cms = evaluate_classifier(predictions,y_HOLDOUT)

#record results
name_arg = 'plus_avg_sentiment'
record_results([predictions,y_HOLDOUT],FILENAME,name_arg)


#scores = [line['sentiment_score'] for line in sentiment_stats_per_article[w]]
#
#print(np.mean([scores[i] for i,x in enumerate(word2context[w]) if labels[x['sdom']]['bias'] in ["L","LC"]]))
#
#print(np.mean([scores[i] for i,x in enumerate(word2context[w]) if labels[x['sdom']]['bias'] in ["R","RC"]]))
#
#print(np.mean(scores))
#
#plt.figure()
#plt.hist([score for i,score in enumerate(scores) if labels[word2context[w][i]['sdom']]['bias'] in ['R']],color='red')
#plt.figure()
#plt.hist([score for i,score in enumerate(scores) if labels[word2context[w][i]['sdom']]['bias'] in ['L']],color='blue')

       
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
