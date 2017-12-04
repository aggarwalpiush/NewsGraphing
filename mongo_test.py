# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:46:42 2017

@author: nfitch3
"""

from pymongo import MongoClient

client = MongoClient('mongodb://gdelt:meidnocEf1@11.7.124.26:27017/')
#Connect to the GDELT Mongo database
#Credentials might be different now, ask David
db = client.gdelt.metadata


print(db.find({},{'text':1,'sourceurl':1}))

#N = 1000000 #10000
#stuff = db.find({},{'links':1,'sourceurl':1}).sort("_id",-1).limit(N)