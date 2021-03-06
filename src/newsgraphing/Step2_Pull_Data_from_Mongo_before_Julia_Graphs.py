# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 10:01:51 2017

@author: nfitch3
"""

from pymongo import MongoClient
import tldextract
#import matplotlib.pyplot as plt
import itertools
import math
import numpy as np
import re
import pickle
import feather
from tqdm import tqdm
import pandas as pd
import pickle
import psycopg2 as pg
import csv




#client = MongoClient('mongodb://gdelt:meidnocEf1@gdeltmongo1:27017/')
client = MongoClient('mongodb://gdelt:meidnocEf1@10.51.4.172:18753/')
db = client.gdelt.metadata

def valid(s, d):
    if  len(d) > 0 and d[0] not in ["/", "#", "{"] and s not in d :
        return True
    else:
        return False


re_3986 = re.compile(r"^(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)(\?([^#]*))?(#(.*))?")
wgo = re.compile("www.")

fulldata = []
stuff = db.find({},{'links':1,'sourceurl':1}).sort("_id",-1)#.limit(10000)
doms = []
print("downloaded!")
for obj in tqdm(stuff):
    if 'links' in obj:
        sdom = re_3986.match(obj['sourceurl']).group(4)
        for link in obj['links']:
            if sdom and valid(sdom, link[0]):
                ddom = re_3986.match(link[0]).group(4)
                if ddom:
                    fulldata.append([wgo.sub("",sdom), wgo.sub("",ddom), link[1]])
                    
                    
df = pd.DataFrame(fulldata, columns=['sdom', 'ddom', 'link'])

feather.write_dataframe(df, "save.feather")

print("Finished successfully!")

