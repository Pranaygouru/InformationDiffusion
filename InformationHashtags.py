'''
Name: Pranay Gouru.
Project: To find an efficient approach to see information flow change over a period of time on twitter data.
Input: Flume files collected from twitter.
Output: Clusters of topics which are hashtags.
Description: A graph with weighted edges to construct a network for information.For this,'Text' and 'full_text'
             columns are parsed from the json.Hashtags and usernames which is 'screenname' are collected from
             the files. Hashtags are considered as nodes for the graph and usernames connecting hashtags as
             weighed edges.The graph is represented as adjacency matrix and spectral clustering method is used
             to model the network. The clusters formed are used to see the information flow change over a period
             of time for the keyword chosen from the data.
'''
# Import statements.
import os
import glob
import json
import re
import scipy.cluster
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
import sklearn
import cmath
from scipy.sparse import csgraph
from sklearn.cluster import KMeans
# Creating dictionaries for hashtags and usernames.
List_Hashtags = {}
usernames = {}
list_user_hashtags = {}
ActualHashtags = {}
ActualUsernames = []
textdict = {}


# All the flume files are clubbed into a single file and reading it
f = open('Trump_data','r',encoding="utf8")
for i in f.readlines():
      # Reading the json file into a dictionary
      Dicta = json.loads(i)
      # Searching for "extended_tweet" column from the json.
      if "extended_tweet" in Dicta.keys():
          # Fetching the hashtags from the full_text field also checking the language english.
          if "#" in Dicta["extended_tweet"]["full_text"] and Dicta["lang"] == "en":
            line = Dicta["extended_tweet"]["full_text"].replace("\n","")
            linetext = line
            # Removing the urls
            line = line.replace("https:"," ")
            arra = line.split('#')
            hashtags = []
            for t in range(1,len(arra)):
                val = arra[t].split(" ")
                arra[t] = val[0]
                hashtags.append(arra[t])
                ltemp = []
                if arra[t] not in textdict:
                    textdict[arra[t]]=ltemp.append(linetext)
                else:
                    ltempp = textdict.get(arra[t])
                    if not ltempp is  None:
                     textdict[arra[t]] = ltempp.append(linetext)
                # Hashtags are in this dictionary
                ActualHashtags[arra[t]] = 0
                # Usernames are collected in to this dictionaries.
                ActualUsernames.append(Dicta["user"]["screen_name"])
                if arra[t] in List_Hashtags.keys():
                    val = List_Hashtags.get(arra[t])
                    val1 = val + 1
                    List_Hashtags[arra[t]] = val1
                else:
                    List_Hashtags[arra[t]] = 1
            usernames[Dicta["user"]["screen_name"]] = hashtags
      #      Fetching the text column to get the hashtags.
      elif "#" in Dicta["text"] and Dicta["lang"] == "en":
           line = Dicta["text"].replace("\n","")
           linetext = line
           line = line.replace("https:"," ")
           arra = line.split('#')
           hashtags = []
           for t in range(1, len(arra)):
               val = arra[t].split(" ")
               arra[t] = val[0]
               hashtags.append(arra[t])
               lt = []
               if arra[t] not in textdict:
                   textdict[arra[t]] = lt.append(linetext)
               else:
                   ltte = textdict.get(arra[t])
                   if not ltte is None:
                    textdict[arra[t]] = ltte.append(linetext)

               ActualHashtags[arra[t]] = 0
               ActualUsernames.append(Dicta["user"]["screen_name"])
               if arra[t] in List_Hashtags.keys():
                   val = List_Hashtags.get(arra[t])
                   val1 = val + 1
                   List_Hashtags[arra[t]] = val1
               else:
                   List_Hashtags[arra[t]] = 1
           usernames[Dicta["user"]["screen_name"]] = hashtags
d = {(k,v) for k,v in List_Hashtags.items()}
ls = list(ActualHashtags.keys())

# Adjacency matrix
adjacency_mat = np.zeros(shape=(np.alen(ls),np.alen(ls)))

# Degree Matrix
degree_mat = np.zeros(shape=(np.alen(ls),np.alen(ls)))
np.fill_diagonal(degree_mat,1)

# Building adjacency matrix and degree matrix.
for key,value in usernames.items():
    if len(value) > 1:
        ind = []
        for i in value:
            ind.append(ls.index(i))
        for k in range(0,len(ind)):
            for p in range(k+1,len(ind)):
                if ind[p]!=ind[k]:
                 adjacency_mat[ind[k]][ind[p]] = adjacency_mat[ind[k]][ind[p]]+1
                 adjacency_mat[ind[p]][ind[k]] = adjacency_mat[ind[p]][ind[k]]+1
                 degree_mat[ind[p]][ind[p]] = degree_mat[ind[p]][ind[p]]+1
                 degree_mat[ind[k]][ind[k]] = degree_mat[ind[k]][ind[k]]+1

sumof = np.sum(adjacency_mat, axis=0)
sumof = sumof.astype(int)
argssumof = sumof.argsort()[-10:][::-1]
hashtagsname = []
hashtagsmentions = []
s=0
for h in argssumof:
    hashtagsname.insert(s,ls[h])
    hashtagsmentions.insert(s,sumof[h])
    s = s+1

# Saving the hashtags into text file.
np.savetxt("hashtags.csv", list(zip(hashtagsname,hashtagsmentions)),fmt='%s')

# Transpose of adjacency matrix(refer: spectral clustering algorithm)
Atrasnpose = np.transpose(adjacency_mat)
# Constructing laplasian matrix (L = D - A).

laplasian = np.subtract(degree_mat,adjacency_mat)

# Finding the eigen values and eigen vectors.
(eigval,eigvect) = np.linalg.eigh(laplasian)
eigval = np.array(eigval)
eigval = eigval.astype(int)

ei = np.argsort(eigval)

#Storing the eigen values into a file.
np.savetxt("eigenvalues.csv", eigval, delimiter=" ")

firstkmat = eigvect[ei[::-1][0:12]]
firstkmat = np.transpose(firstkmat)

# K means++ is used and the number of clusters are chosen from the eigen value plot.

kmeans = KMeans(n_clusters = np.alen(firstkmat[0]),init='k-means++',max_iter=300,precompute_distances = True)
kmeans.fit(firstkmat)
labels = kmeans.predict(firstkmat)
vald = 0
labelsdict = {}
for i in labels:
 if labelsdict.get(i) == None:
  labelsdict[i]=[]
 labelsdict[i].append(ls[vald])
 vald = vald + 1
eivv = np.where(eigval == 0)[0]
np.savetxt('eigen.csv', sumof, delimiter=',')

#Collecting the clusters of hashtags into a text file.

w = csv.writer(open("dictlabels.txt","w"))
w1 = csv.writer(open("dictlab.txt","w"))
for ke,va in labelsdict.items():
 w.writerow([ke,va])
 for h in va:
  if not textdict.get(h) is None:
   w1.writerow(textdict.get(h))



