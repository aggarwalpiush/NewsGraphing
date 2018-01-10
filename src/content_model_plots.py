# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 09:25:11 2017

@author: nfitch3
"""


from sklearn import metrics
import matplotlib.pyplot as plt
import csv
import os
import sys
from collections import Counter

CLF_ARGS = ['logreg','rf','svm']
LABEL_ARGS = ['bias','cred']
PATH = '../results/'
# files should have format: CLFNAME_LABELNAME_OPTION_results.csv
files = ['logreg_bias_results.csv','rf_bias_results.csv','logreg_cred_results.csv','logreg_bias_paragraph_vectors_results.csv']


def makeROC(data,args):
    
    roc_data = {}
    for i,pair in enumerate(data):
        CLFNAME = args[i][0]
        LABELNAME = args[i][1]
        
        if len(args[i])>2:
            OPTION = '('+'-'.join(args[i][2:])+')'
        else:
            OPTION = '(tf-idf)'
        
        if LABELNAME not in roc_data:
            roc_data[LABELNAME] = []
        
        # legend info
        roc_label = CLFNAME + ' ' + OPTION
        
        # compute AUC
        predictions = [p[0] for p in pair]
        truth = [p[1] for p in pair]
        fpr, tpr, roc_thresholds = metrics.roc_curve(truth,predictions)
        auc = metrics.auc(fpr,tpr)
        print("AUC score for "+CLFNAME+" "+LABELNAME+": ", auc)
        
        # plot type and title
        if LABELNAME == 'bias':
            title = 'Bias Classification Receiver Operating Characteristic'
        elif LABELNAME == 'cred':
            title = 'Credibility Classification Receiver Operating Characteristic'
        
        # save ROC data
        roc_data[LABELNAME].append([[fpr,tpr],roc_label,title,auc])
      
    rocs = []
    for label in roc_data:
        # new plot
        plt.figure()

        # plot random line
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') 
        for dataset in roc_data[label]: 
            # extract data
            fpr = dataset[0][0]
            tpr = dataset[0][1]
            roc_label = dataset[1]
            title = dataset[2]
            auc = dataset[3]          
            # plot ROC Curve for each dataset with label
            plt.plot(fpr,tpr,lw=2, label=roc_label + '(area = %0.3f)' % auc)

        plt.title(title)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.savefig(PATH+'content_model_'+label+'_roc.png',bbox_inches='tight')

    return rocs

def distribution_by_label(predictions,truth):
    
    # Histogram showing overlap of predicted labels
    plt.hist([x for i,x in enumerate(predictions) if truth[i]==0])
    plt.hist([x for i,x in enumerate(predictions) if truth[i]==1])
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.title('Prediction Distribution by Label')
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.savefig(PATH+'content_model_predictions_distribution.png',bbox_inches='tight')
    plt.show()
    
    return plt



# main
args = []
data = []
# extract parameters from file names
for file in files:
    if not os.path.exists(PATH+file):
        sys.exit("One or more input files do not exist: " + str(files))
        
    args.append(file.split('_')[:-1])
    file_data = []
    # read in prediction/truth data
    with open(PATH+file,'r',encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) #skip header
        for row in reader:
            file_data.append([float(i) for i in row])
    data.append(file_data)

# create and save ROC curve
figs = makeROC(data,args)


## create and save histogram of predicted labels
#plot = distribution_by_label()
