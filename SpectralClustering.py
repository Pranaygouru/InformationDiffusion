import os
import glob
import json
import re
import scipy.cluster
from sklearn.cluster import SpectralClustering
import csv
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category = DeprecationWarning)
import numpy as np
import sklearn
import cmath
from scipy.sparse import csgraph
from sklearn.cluster import KMeans
from scipy import linalg as LA
List_Hashtags = {}
usernames = {}
list_user_hashtags = {}
ActualHashtags = {}
ActualUsernames = []
textdict = {}
# Local Path
# k = 3
path = '/Users/prana/PycharmProjects/IndeStudy/venv/'
# Reading flume files
def transformToSpectral(laplacian):
    global k
    e_vals, e_vecs = LA.eig(np.matrix(laplacian))
    ind = e_vals.real.argsort()[:2]
    result = np.ndarray(shape=(laplacian.shape[0],0))
    for i in range(1, ind.shape[0]):
        cor_e_vec = np.transpose(np.matrix(e_vecs[:,np.asscalar(ind[i])]))
        result = np.concatenate((result, cor_e_vec), axis=1)
    return result
for filename in glob.glob(os.path.join(path, 'Flume*')):
  # Opening File
  f = open(filename,'r',encoding="utf8")
  for i in f.readlines():
      Dicta = json.loads(i)
      # print(Dicta.keys())
      # print(Dicta["created_at"])
      # print(Dicta["user"]["screen_name"])
      # If it has a extended tweet of which full text is read
      if "extended_tweet" in Dicta.keys():
          if "#" in Dicta["extended_tweet"]["full_text"] and Dicta["lang"] == "en":
              # Data Cleaning
              line = Dicta["extended_tweet"]["full_text"].replace("\n", "")
              linetext = line
              # Data Cleaning
              line = line.replace("https:", " ")
              # splitting the line with hash tags into an array.
              arra = line.split('#')
              # print(Dicta["user"]["screen_name"])
              hashtags = []
              # print(arra)
              for t in range(1, len(arra)):
                  val = arra[t].split(" ")
                  arra[t] = val[0]
                  result = re.search("^[a-zA-Z0-9_.-]*$",val[0])
                  if result is not None:
                   hashtags.append(arra[t])
                  ltemp = []
                  if arra[t] not in textdict:
                      textdict[arra[t]] = ltemp.append(linetext)
                  else:
                      ltempp = textdict.get(arra[t])
                      if not ltempp is None:
                          textdict[arra[t]] = ltempp.append(linetext)
                  ActualHashtags[arra[t]] = 0
                  ActualUsernames.append(Dicta["user"]["screen_name"])
                  # print(Dicta["user"]["screen_name"])
                  if arra[t] in List_Hashtags.keys():
                      val = List_Hashtags.get(arra[t])
                      val1 = val + 1
                      List_Hashtags[arra[t]] = val1
                  else:
                      List_Hashtags[arra[t]] = 1
              usernames[Dicta["user"]["screen_name"]] = hashtags
              # if matchobj:
              #     print(matchobj.group())
      elif "#" in Dicta["text"] and Dicta["lang"] == "en":
          line = Dicta["text"].replace("\n", "")
          linetext = line
          line = line.replace("https:", " ")
          # print(Dicta["user"]["screen_name"])
          # # matchobj = re.match((".*#[a-z][ ]"), line, re.M | re.I)
          arra = line.split('#')
          hashtags = []
          for t in range(1, len(arra)):
              val = arra[t].split(" ")
              result = re.search("^[a-zA-Z0-9_.-]*$", val[0])
              if result is not None:
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
  d = {(k, v) for k, v in List_Hashtags.items()}
  # print(d)
  # print(np.unique(ActualHashtags))
  ls = list(ActualHashtags.keys())
  # print(ActualUsernames)
  adjacency_mat = np.zeros(shape = (np.alen(ls), np.alen(ls)))
  degree_mat = np.zeros(shape = (np.alen(ls), np.alen(ls)))
  np.fill_diagonal(degree_mat, 1)
  # print(adjacency_mat)
  # print(degree_mat)
  Hashtags_Users_list = {}
  for key, value in usernames.items():
      if len(value) > 1:
          ind = []
          for i in value:
              # print(i)
              ind.append(ls.index(i))
          for k in range(0, len(ind)):
              for p in range(k + 1, len(ind)):
                  if ind[p] != ind[k]:
                      adjacency_mat[ind[k]][ind[p]] = adjacency_mat[ind[k]][ind[p]] + 1
                      adjacency_mat[ind[p]][ind[k]] = adjacency_mat[ind[p]][ind[k]] + 1
                      degree_mat[ind[p]][ind[p]] = degree_mat[ind[p]][ind[p]] + 1
                      degree_mat[ind[k]][ind[k]] = degree_mat[ind[k]][ind[k]] + 1

Atrasnpose = np.transpose(adjacency_mat)
laplasian = np.subtract(degree_mat, adjacency_mat)
transformedData = transformToSpectral(laplasian)
# print(transformedData)
# kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, precompute_distances=True)
# kmeans.fit(transformedData)
# labels = kmeans.predict(transformedData)
# print(labels)


