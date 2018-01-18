# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 10:01:51 2017

@author: nfitch3
"""

import argparse
import os.path
# import tldextract
import logging
import re

import feather
import pandas as pd
from pymongo import MongoClient

PROGRESS = False
try:
    from tqdm import tqdm
except ImportError:
    logging.warning('tqdm is not installed, progress meters will not be available.')
    PROGRESS = True

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

DEFAULTCONN = 'mongodb://gdelt:meidnocEf1@10.51.4.177:20884/'
re_3986 = re.compile(r"^(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)(\?([^#]*))?(#(.*))?")
wgo = re.compile("www.")


def connect(connstring):
    """
    connect(connstring) connects to a mongodb instance using a single connections string
    """
    client = MongoClient(connstring)
    logging.info(client.admin.command('ismaster'))
    db = client.gdelt.metadata
    return client, db


def valid(s, d):
    if len(d) > 0 and d[0] not in ["/", "#", "{"] and s not in d:
        return True
    else:
        return False


def filetype(path):
    """determing the filetype of a path from its extension"""
    if os.path.splitext(path)[1] == '.feather':
        return 'feather'
    elif os.path.splitext(path)[1] == '.csv':
        return 'csv'
    raise(Exception('{} is neither a feather nor csv file'.format(path)))


def extract_domain(url):
    return re_3986.match(url).group(4)


def collect_links(db, query, limit, quiet=False):
    """collect_links takes a mongodb connection and a limitand pulls all the links out of the mongo.

    limit: the number of records to pull: -1 for all of them

    uses tqdm to show progress.
    """
    N = limit
    fulldata = []
    print("using a limit of {}".format(N))

    stuff = db.find({}, query).sort("_id", -1).limit(N)
    # stuff = db.find().sort("_id",-1).limit(N)
    print("downloaded!")
    iterations = stuff
    if not quiet and PROGRESS:
        iterations = tqdm(stuff)
    for obj in iterations:
        if 'links' in obj:
            sdom = extract_domain(obj['sourceurl'])
            if not sdom:
                continue
            for link in obj['links']:
                if valid(sdom, link[0]):
                    ddom = extract_domain(link[0])
                    if ddom:
                        linktype = link[1]
                        row = [wgo.sub("", sdom), wgo.sub("", ddom), linktype]
                        fulldata.append(row)
    return fulldata


def parse_arguments():
    desc = 'Pull down the data from a mongodb to a static file.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('limit', metavar='N', type=int,
                        help='Limit on the number of articles -1 for all of them')

    parser.add_argument('-o', '--outfile', default='save.feather', type=str,
                        help='path to store the resulting data')

    parser.add_argument('-c', '--connection', type=str,
                        default=DEFAULTCONN, help='The mongodb connections string')
    parser.add_argument('-q', '--quiet', action='store_true',
                        default=DEFAULTCONN, help='The mongodb connections string')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    N = args.limit
    outfile = args.outfile
    conn = args.connection
    quiet = args.quiet
    query = {'links': 1, 'sourceurl': 1}
    client, db = connect(conn)
    fulldata = collect_links(db, query, N, quiet)

    # df = pd.DataFrame(fulldata, columns=['sdom article ID','sdom', 'ddom', 'link'])
    df = pd.DataFrame(fulldata, columns=['sdom', 'ddom', 'link'])

    ftp = filetype(outfile)
    if ftp == 'feather':
        feather.write_dataframe(df, outfile)
    else:
        df.to_csv(outfile)
