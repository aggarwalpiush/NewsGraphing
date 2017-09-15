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

