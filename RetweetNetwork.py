'''
Similar to what we have constructed in the informationhashtags file but here
connections are retweets and nodes are usernames.
'''
# Import statements
import os
import glob
import json
import re
import scipy.cluster
import csv
import warnings
import json,sys,re,os
from difflib import SequenceMatcher
from datetime import timedelta, date
import time, datetime
from sklearn.cluster import SpectralClustering
warnings.filterwarnings("ignore", category = DeprecationWarning)
import numpy as np
import sklearn
import cmath
from scipy.sparse import csgraph
from sklearn.cluster import KMeans
import networkx as nx
import matplotlib.pyplot as plt

# Reading the path of flume files

# This function removes new line & return carriages from tweets
def stripNewLineAndReturnCarriage(tweetText):
    return tweetText.replace('\n', ' ').replace('\r', '').strip().lstrip()
# This function is used to remove user mentions in a tweet
def removeUserMentions(tweetText):
    return re.sub('@[^\s]+','',tweetText)
# This function replaces words with repeating 'n' or more same characters with a single character
def replaceRepeatedCharacters(tweetText):
    return re.sub(r"(.)\1{3,}",r"\1", tweetText)
# This function is used to convert multiple white spaces into a single white space
def convertMultipleWhiteSpacesToSingleWhiteSpace(tweetText):
    return re.sub('[\s]+', ' ', tweetText)
# This function replaces any hash tag in a tweet with the word
def replaceHashTagsWithWords (tweetText):
    return re.sub(r'#([^\s]+)', r'\1', tweetText)
def textCleaning(tweet_text):
    if tweet_text != "":
        tweet_text = stripNewLineAndReturnCarriage(tweet_text)
        tweet_text = removeUserMentions(tweet_text)
        tweet_text = replaceRepeatedCharacters(tweet_text)
        tweet_text = convertMultipleWhiteSpacesToSingleWhiteSpace(tweet_text)
        tweet_text = replaceHashTagsWithWords(tweet_text)
        tweet_text = re.sub(r'([^\s\w:./]|_)', '', tweet_text)
        tweet_text = tweet_text.strip().lstrip().lower()
        tweet_text = re.sub(r'http\S+', '', tweet_text)
        if tweet_text[0:2] == "rt":
            tweet_text = tweet_text[3:].lstrip()
        return tweet_text

tweet_text_set = {""}
tweet_text_Users = {}
# Reading the flume files line by line
f = open('Trump_data','r',encoding="utf8")
for i in f.readlines():
    Dicta = json.loads(i)
    #Collecting the retweets
if 'retweeted_status' in Dicta and Dicta["lang"] == "en":
    this_tweet_timestamp = Dicta['created_at'].lstrip().strip()

    this_user_handle = Dicta['user']['screen_name'].lstrip().strip()
    if "extended_tweet" in Dicta.keys():
        if Dicta["extended_tweet"]["full_text"] and Dicta["lang"] == "en":
            tweet_text = Dicta["extended_tweet"]["full_text"]
    else:
        tweet_text = Dicta['text'].lstrip().strip()
    tweet_text = textCleaning(tweet_text)
    if tweet_text != "":
        flag = 0
        if tweet_text in tweet_text_set:
            tempList = []
            tempList = tweet_text_Users.get(tweet_text)

            tempList.append(this_user_handle)
            tweet_text_Users[tweet_text] = tempList
            flag = 1
        if flag != 1:
            # Finding the match for sequence to add edges
            for i in tweet_text_set:
                s = SequenceMatcher(None, tweet_text, i).ratio()
                if s > 0.95:
                    tempList = []
                    tempList = tweet_text_Users.get(tweet_text)

                    if tempList is not None:
                        tempList.append(this_user_handle)
                        tweet_text_Users[tweet_text] = tempList
                        flag = 1
    if flag != 1:
        if tweet_text not in tweet_text_set:
            tweet_text_set.add(tweet_text)
            list_of_users = []
            list_of_users.append(this_user_handle)
            tweet_text_Users[tweet_text] = list_of_users

    if 'retweeted_status' in Dicta and Dicta["lang"] == "en":
        #Text cleaning and collecting the data.
        retweetText = Dicta['retweeted_status']['text']
        retweetText = textCleaning(retweetText)

        ownerName = Dicta['retweeted_status']['user'][
            'screen_name'].lstrip().strip()
        ownerTimeStamp = Dicta['retweeted_status'][
            'created_at'].lstrip().strip()

lis = {}
lis = tweet_text_Users.values()
username_list_dict = {}
count = 0
for username_list in lis:
    if len(username_list) > 1:
        for un in username_list:
            if un not in username_list_dict:
                username_list_dict[un] = count
                count = count + 1

# Adjacency matrix
array_for_adj = np.zeros((count,count),dtype=int)
array_for_deg = np.zeros((count,count),dtype=int)
np.fill_diagonal(array_for_deg,1)

# Constructing adjacency matrix and degree matrix
for username_list in lis:
    if len(username_list) > 1:

        Parenttweet_user = username_list[0]

    for parselist in range(1,len(username_list)):

        array_for_deg[username_list_dict.get(Parenttweet_user)][username_list_dict.get(Parenttweet_user)] = \
            array_for_deg[username_list_dict.get(Parenttweet_user)][username_list_dict.get(Parenttweet_user)] + 1
        array_for_deg[username_list_dict.get(username_list[parselist])][username_list_dict.get(username_list[parselist])] = \
            array_for_deg[username_list_dict.get(username_list[parselist])][username_list_dict.get(username_list[parselist])] + 1
        array_for_adj[username_list_dict.get(Parenttweet_user)][username_list_dict.get(username_list[parselist])] = \
            array_for_adj[username_list_dict.get(Parenttweet_user)][username_list_dict.get(username_list[parselist])] + 1

        array_for_adj[username_list_dict.get(username_list[parselist])][username_list_dict.get(Parenttweet_user)] = \
            array_for_adj[username_list_dict.get(username_list[parselist])][username_list_dict.get(Parenttweet_user)] + 1

# Transpose of a adjacency matrix as part of spectral clustering algorithm.

Atrasnpose = np.transpose(adjacency_mat)
laplasian = np.subtract(degree_mat,adjacency_mat)

(eigval,eigvect) = np.linalg.eigh(laplasian)
eigval = np.array(eigval)
eigval = eigval.astype(int)

ei = np.argsort(eigval)
# Saving the eigen values into a text file.
np.savetxt("eigenvalues.csv", eigval, delimiter=" ")
G = nx.from_numpy_matrix(eigvect)
nx.draw_networkx(G,with_labels=True)

firstkmat = eigvect[ei[::-1][0:4]]
firstkmat = np.transpose(firstkmat)

# Clustering using kmeans++ and the number of clusters are chosen from the eigen value plot.
kmeans = KMeans(n_clusters=np.alen(firstkmat[0]), init='k-means++', max_iter=100, precompute_distances=True)

kmeans.fit(firstkmat)
labels = kmeans.predict(firstkmat)

clusters_dict = {}
def get_key(val):
    for key, value in username_list_dict.items():
         if val == value:
             return key
for i in range(0,len(labels)):
    ls = clusters_dict[labels[i]]
    if labels[i] not in  clusters_dict:
        clusters_dict[labels[i]]=[]
    ls = []
    if not len(clusters_dict.get(labels[i]))==0 :
     ls = clusters_dict.get(labels[i])
     clusters_dict[labels[i]]=ls.append(get_key(i))
    else:
     ls = ls.insert(0,get_key(i))
     clusters_dict[labels[i]].append(ls)

    if clusters_dict.get(labels[i]) == None:
        clusters_dict[labels[i]] = []
    clusters_dict[labels[i]].append(get_key(i))


