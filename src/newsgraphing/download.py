# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 13:35:52 2018

@author: nfitch3
"""

import requests

DATADIR = '../../data'
urls = ['https://www.dropbox.com/s/jzi8lm9hkntwkga/save.feather?dl=1','https://www.dropbox.com/s/ct3j5mjz7sdwo9x/gdelt_text.csv?dl=1','https://www.dropbox.com/s/4ywhh7giljarzlz/bias.csv?dl=1']
local_filenames = ['save.feather','gdelt_text.csv','bias.csv']
for i,url in enumerate(urls):
    r = requests.get(url, stream=True)
    with open(DATADIR + '/' +  local_filenames[i], 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
