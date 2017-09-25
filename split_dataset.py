# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 10:56:08 2017

@author: nfitch3
"""

import random
import os
from collections import defaultdict


def generate_hold_out_split (dataset, training = 0.8, base_dir="splits"):
    r = random.Random()
    r.seed(1489215)

    domains = dataset.keys()  # get list of domain names
    r.shuffle(domains)  # and shuffle that list


    training_set = domains[:int(training * len(domains))]
    hold_out_set = domains[int(training * len(domains)):]

    # write the split body ids out to files for future use
#    with open(base_dir+ "/"+ "training_ids.txt", "w+") as f:
#        f.write("\n".join([str(id) for id in training_set]))
#
#    with open(base_dir+ "/"+ "hold_out_ids.txt", "w+") as f:
#        f.write("\n".join([str(id) for id in hold_out_set]))
        
    return training_set,hold_out_set