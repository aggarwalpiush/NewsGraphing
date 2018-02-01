# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 14:15:52 2017

@author: nfitch3
"""
from pymongo import MongoClient
import re
import csv
from download import downloadfiles
import os
import sys
from content_model_text_functions import clean
import logging
from collections import Counter

logging.basicConfig(format='%(levelname)s %(asctime)-15s %(message)s', level=logging.INFO) 

defaultmgoclient = MongoClient('mongodb://gdelt:meidnocEf1@10.51.4.177:20884/')
re_3986 = re.compile(r"^(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)(\?([^#]*))?(#(.*))?")
wgo = re.compile("www.")

def processdatafiles(FILEPATH,CLEANFILEPATH,BIASFILE,DATADIR):
    """Download and read in all the text csvfiles for the content model"""
     
    # check if csv files already exist (gdelt_text.csv, bias.csv)
    if not (os.path.exists(FILEPATH)) or not (os.path.exists(BIASFILE)):
        logging.info("Downloading data files from dropbox")
        downloadfiles(DATADIR)
        
    # read in gdelt article text
    logging.info("Reading data from cache")
    corpus, articles, i2s, sdom_counts = readtextcsv(FILEPATH,DATADIR)
    
    # check if gdelt_text_clean.csv already exists
    if not (os.path.exists(CLEANFILEPATH)):
        logging.info("Writing out cleaned text corpus")
        clean_corpus = writecleancsv(clean, CLEANFILEPATH, corpus, i2s)
    else:
        logging.info("Reading (clean) data from cache")
        clean_corpus,_,_,_ = readtextcsv(CLEANFILEPATH,DATADIR)
    
    # read in scraped MBFC labels
    logging.info("Reading in scraped MBFC data labels")
    labels,biasnames = readbiasfile(BIASFILE)
    
    print('Number of domains with at least one label', len(set(biasnames)))
    
    return corpus, clean_corpus, articles, i2s, labels, sdom_counts

def readtextcsv(filename,DATADIR):
    """Read in the text csv from filename.

    filename: the file from which to read
    DATADIR: the dir to which to write out
    
    corpus: a list of documents
    articles: a dict mapping source domain to its text
    i2s: a dict mapping article index to source domain for the articles in the corpus
    sdom_counts: a dict mapping source domain to its number of articles in the corpus

    """
    
    # read in/format data files
    maxInt = sys.maxsize
    decrement = True   
    articles = {}
    i2s = {}
    corpus = []
    sdoms = []
    with open(filename, 'r',encoding='utf-8') as csvfile:
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
            
            sdoms.append(row[1])
            
            # Create the mapping from article ID to sdom name
            i2s[i] = row[1]

    sdom_counts = Counter(sdoms)    
    if not (os.path.exists(os.path.join(DATADIR, 'sdom_by_article.csv'))):
        with open(os.path.join(DATADIR, 'sdom_by_article.csv'), "w",encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(["sdom","number of articles"])
            for key in sdom_counts.keys():
                # Write rows
                writer.writerow([key,sdom_counts[key]])
            
    return corpus, articles, i2s, sdom_counts
        

def readbiasfile(filename):
    """This opens up the MBFC labels which were scraped off their website"""

    biasnames = []
    # Political Bias
    pol = ['L', 'LC', 'C', 'RC', 'R']
    # Reporting Quality
    rep = ['VERY LOW', 'LOW', 'MIXED', 'HIGH', 'VERY HIGH']
    # Fake categories: Fake, Conspiracy, Satire
    flag = ['F', 'X', 'S']

    labels = {}
    
    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            url = re_3986.match(row[4]).group(4)
            if url:
                name = wgo.sub("", url)
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

                if ( labels[name]['bias'] != 'na' or  labels[name]['cred'] != 'na') and name not in biasnames:
                    biasnames.append(name)

    return labels, biasnames


def sourcedomain(obj):
    """parse out the domain of the sourcurl field of a mongo object"""
    wgo = re.compile("www.")
    return wgo.sub("", re_3986.match(obj['sourceurl']).group(4))


def get_article_text(biasfile, mgoclient=defaultmgoclient, sample=1000):
    """download articles from the mongo client and for return the text for all the domains
    that are labeled according to biasfile."""
    # Connect to the GDELT Mongo database
    db = mgoclient.gdelt.metadata
    # Regular Expression to process web domains into chunks
    # For replacing www.
    # whitelist = ["NOUN", "PROPN", "ADJ", "ADV"]
    # Types of words we'll look at
    labels, biasnames = readbiasfile(biasfile)
    biasnameset = set(biasnames)
    stuff = db.find({}, {'text': 1, 'sourceurl': 1}).sort("_id", -1).limit(sample)
    arts = {}
    counts = {}
    print('Number of domains with at least one label', len(set(biasnames)))

    i2s = {}
    article_id_count = 0
    corpus = []
    # Download articles and process them with SpaCy
    for obj in stuff:
        if 'text' in obj:
            sdom = sourcedomain(obj)
            if sdom in biasnameset:
                # doc = nlp.tokenizer(obj['text'][:100*8])
                # nlp.tagger(doc)
                # Only break into tokens and give them part of speech tags
                # doc_sentences = [sent.string.strip() for sent in doc.sents]
                doc = ' '.join(obj['text'].split())
                if sdom not in arts.keys():
                    arts[sdom] = []
                    counts[sdom] = 0
                # Create article ID to sdom map
                i2s[article_id_count] = sdom
                article_id_count += 1
                # Create corpus (each article is a document)
                corpus.append(doc)
                arts[sdom].append(doc)
                counts[sdom] += 1

    print('Number of domains with text and at least one label', len(arts.keys()))
    print('Total number of articles: ', sum(counts.values()))

    return (arts, corpus, labels, i2s)


def writetextcsv(file_name, corpus, i2s):
    """Clean up the text from the articles."""
    with open(file_name, "w", encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["article_id", "sdom", "text"])
        # Concatenate each domain's text into corpus
        for i, article in enumerate(corpus):
            # Write rows
            sdom = i2s[i]
            writer.writerow([i, sdom, article])


def writecleancsv(cleanfunc, cleanfile, corpus, i2s):
    """write out the cleaned up text into to csv file.

    cleanfunc: a function that takes a row of text and cleans it up.
    cleanfile: the file in which to store the output
    corpus: a list of documents
    i2s: a map from index to label for the docs in the corpus.

    """
    clean_corpus = []
    with open(cleanfile, "w", encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["article_id", "sdom", "text"])
        for i, article in enumerate(corpus):
            # Write rows
            sdom = i2s[i]
            # Clean corpus along the way
            clean_article = cleanfunc(article)
            writer.writerow([i, sdom, clean_article])

            # Create cleaned corpus
            clean_corpus.append(clean_article)
    return clean_corpus
