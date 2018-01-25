# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 13:12:52 2017

@author: nfitch3
"""



from __future__ import unicode_literals, print_function

import string
from sklearn import feature_extraction
#from nltk.stem import PorterStemmer,SnowballStemmer
#import nltk
#nltk.download('punkt')
from nltk import tokenize, word_tokenize, sent_tokenize
import spacy

#from gensim.models.doc2vec import Doc2Vec
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as SIA


nlp = spacy.load('en_core_web_sm')


def find_subjects(myString):
    """ Find and return word(s) in input string that are the subject(s)
    
    Arguments
    - myString: a string of text

    Returns:
    - sub_toks: a list of subject words
        
    """
    
    # use spacy to find subject in sentence
    doc = nlp(myString)
    
    #nouns = [i for i in doc.noun_chunks]
    sub_toks = [str(word) for word in doc if (word.dep_ == "nsubj") ]
    
    return sub_toks

def text_to_vector(model, text):
    """Create paragraph vector from input string
    
    Arguments
    - text: a string of text
    - model: a trained Doc2Vec model

    Returns:
    - text_vector: a 1x300 dimensional vector representing the input text
        
    """
    
    text_words = remove_stop_words(word_tokenize(text))
    model.random.seed(0)
    text_vector = model.infer_vector(text_words)
    
    return text_vector


def remove_punctuation(myString):
    """Remove punction from input string
    
    Arguments
    - myString: a string of text

    Returns:
    - newString: a string of text without punctuation
        
    """
    
    translator = str.maketrans('', '', string.punctuation)
    newString = myString.translate(translator)
    
    return newString 

def remove_stop_words(tokens):
    """Remove stop words from tokenized list of words
    
    Arguments
    - tokens: a list of words

    Returns:
    - newTokens: a list of words that are not stop words
        
    """
    newTokens = [w for w in tokens if w not in feature_extraction.text.ENGLISH_STOP_WORDS]
    
    return newTokens

def stem_words(myString):
    """Stem words from input string
    
    Arguments
    - myString: a string of text

    Returns:
    - stemmed: a list of stemmed words
        
    """
    
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


def return_top_k_keys(myDict,k):
    """Returns 'k' keys with largest values (i.e. counts)
    
    Arguments
    - myDict: a dict of keys and values
    - k: an integer specifying the number of entries to be returned

    Returns:
    - top_k_keys: a list of keys from myDict
        
    """
    
    top_k_keys = sorted(myDict, key=myDict.get, reverse=True)[:k]
    
    return top_k_keys


def remove_invalid_chars(myString):
    """Remove invalid characters from input string
    
    Arguments
    - myString: a string of text

    Returns:
    - newString: a string with invalid characters replaced
        
    """
    
    newString = myString.replace('“',' ').replace('”',' ').replace('’',"'").replace('©','copyright')
    
    return newString

def clean(myString):
    """Clean input string through pipeline of tasks
    
    Arguments
    - myString: a string of text

    Returns:
    - myString_cleaned: a string of text cleaned according to the specified pipeline
        
    """
    
    no_invalid_chars = remove_invalid_chars(myString)
    no_contractions = expand_contractions(no_invalid_chars.lower())
    no_punctuation = remove_punctuation(no_contractions)
    tokens = no_punctuation.split()
    no_stops = remove_stop_words(tokens)
#    stemmed_tokens = stem_words(' '.join(no_stops))
#    no_numbers = []
#    for token in stemmed_tokens:
#        if not any(char.isdigit() for char in token):
#            no_numbers.append(token)
#    
#    myString_cleaned = " ".join(no_numbers)
    
    myString_cleaned = ' '.join(no_stops)

    return myString_cleaned
       
def expand_contractions(myString):
    """Expand contractions in input string
    
    english_contractions source: https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
    
    Arguments
    - myString: a string of text

    Returns:
    - newString: a string of text with contractions expanded
        
    """
    
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
            
    newString = " ".join(new_s)
    
    return newString


def create_context(contextWord,myCorpus,i2s):
    """Returns sentences in corpus for which given context word appears
    
    Arguments
    - contextWord: a string representing the word/phrase for which you want context
    - myCorpus: a list of article text in the corpus
    - i2s: a dict mapping integer vertex labels to string representations (domain names)

    Returns:
    - all_context: a list containing of dict of relevant sentences their metadata (source domain, article ID, term freq)
        
    """
    
    print('creating context for top word '+str(contextWord))
    all_context = []
    for i,raw_text in enumerate(myCorpus):
        context = {}
        # Adjust raw text
        text = expand_contractions(remove_invalid_chars(raw_text).lower())

        if contextWord in text:
            # grab source its from
            context['sdom'] = i2s[i]
            context['article_ID'] = i
            # split into sentences, remove punctuation, remove stop words
            sentences = sent_tokenize(text)
            sentences = [remove_punctuation(s) for s in sentences]
            #sentences = [' '.join(remove_stop_words(word_tokenize(s))) for s in sentences]
            #            indices = [i for i,x in enumerate(tokens) if x == contextWord]
            # grab sentences in which contextWord appears
            sentences = [sentence for sentence in sentences if contextWord in sentence] #i<len(tokens)-1 and contextWord in tokens[i]+' '+tokens[i+1]]
            context['tf'] = len(sentences)
            
#            # filter sentences where contextWord is the subject
#            keep_sentences = []
#            for sentence in sentences:
#                subjects = find_subjects(sentence)
#                subjects = ['test']
#                if contextWord in subjects:
#                    keep_sentences.append(sentence)
#                    
#            context['sentences'] = keep_sentences

            context['sentences'] = sentences

            all_context.append(context)
    
    return all_context


# Generate topics
#from gensim import corpora, models
#topic_dict = corpora.Dictionary(word2context[w])
#topic_corpus = [topic_dict.doc2bow(text) for text in word2context[w]]
#LDA_model = gensim.models.ldamodel.Ldamodel(topic_corpus,num_topics=2,id2word=topic_dict,passes=20)


