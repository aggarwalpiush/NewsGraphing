# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 10:03:23 2017

@author: nfitch3
"""

from pymongo import MongoClient
import tldextract
import matplotlib.pyplot as plt
import networkx as nx
#import pygraphviz
from networkx.drawing.nx_agraph import graphviz_layout
from scipy.stats import norm
import itertools
import math
import numpy as np
import re
import pickle
import csv
import feather
from tqdm import tqdm
import json


re_3986 = re.compile(r"^(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)(\?([^#]*))?(#(.*))?")

bias = []
biasnames = []
pol = ['L', 'LC', 'C', 'RC', 'R']
rep = ['VERY LOW', 'LOW', 'MIXED', 'HIGH', 'VERY HIGH']
flag = ['F', 'X', 'S']
with open('bias.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        name = re_3986.match(row[4]).group(4)
        p = -1
        r = -1
        f = -1
        if row[1] in pol:
            p = pol.index(row[1])
        if row[2] in rep:
            r = rep.index(row[2])
        if row[3] in flag:
            f = flag.index(row[3])
        bias.append(row + [name, p, r, f, 1 if p == -1 else 0])
        biasnames.append(name)
#print(bias)


with open('save.json') as data_file:    
    counts = json.load(data_file)

domains = set()

for cat in counts: #for each type of tag: a, img, script, link
    for src in tqdm(counts[cat]):
        if src not in domains:
            domains.add(src)
        for dst in counts[cat]:
            if dst not in domains:
                domains.add(dst)
                
    
labels = pol + rep + flag
c = [0]*13 #counts for each label in "labels": ['L', 'LC', 'C', 'RC', 'R', 'VERY LOW', 'LOW', 'MIXED', 'HIGH', 'VERY HIGH', 'F', 'X', 'S']
cm = np.zeros((3, 5)) #confusion matrix: number of categories (political bias, credibility, special label X,S, or F) by number of political bias labels
l = []
for x in range(13):
    l.append([])

for dom in domains:
    if dom in biasnames:
        datum = bias[biasnames.index(dom)][1:4]
        for i in range(3):
            if datum[i] in labels:
                c[labels.index(datum[i])] += 1
                l[labels.index(datum[i])].append(dom)
            if datum[1] in rep and datum[0] in pol:
                cm[4-rep.index(datum[1]),pol.index(datum[0])] += 1
print(cm)
plt.imshow(cm/np.sum(cm,axis=1)[:, np.newaxis], interpolation="nearest")
plt.show()
         
