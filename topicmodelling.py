import os
import glob
import json
import re
import numpy as np
import csv
import textmining
import string
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from joblib import Parallel,delayed
import gensim
from gensim.utils import simple_preprocess
import lda
from sklearn.decomposition import NMF
import enchant
sw = set(stopwords.words('english'))
tdm= textmining.TermDocumentMatrix()
transab = " ".maketrans('','',string.punctuation)
# unparsed=set()
folder = "C:\\Users\\Pranay Gouru\\PycharmProjects\\IndeStudy\\venv\\"
files = [val for sublist in [[os.path.join(i[0],j) for j in i[2]] for i in os.walk(folder)] for val in sublist if 'Flum' in val]
def clean(line):
    # regex = re.compile(r'http*',re.IGNORECASE)
    # line = regex.sub(' ',line)
    # reg = re.compile(r'//*')
    # line = reg.sub(' ',line)
    # line = re.match(r'https+','')
    # line = line.split('https')
    # line = line[0]
    # line = line.split("#")
    # line = line[0]
    line = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", line).split())
    word_tokens = word_tokenize(line)
    line1 = [w for w in word_tokens if not w in sw]
    # line1 = gensim.utils.simple_preprocess(line1,deacc=True,min_len=2)
    str = ' '.join(line1)
    line = str
    if line[:2]=='RT':
        line = line[2:]
    l1 = [x.lower().translate(transab) if x.lower() not in sw else ' ' for x in line.split()]
    ll = [' '.join(e for e in x if e.isalnum()) for x in l1 if x is not "" and len(x.strip()) > 2 and not x.isdigit() and 'https' not  in x.lower()]
    l1 = l1[:np.alen(l1)-1]
    # print(l1)
    print("-----------------------------------")
    print(' '.join(l1))
    # l1 = gensim.utils.simple_preprocess(l1,deacc=True,min_len=2)
    str = ' '.join(l1).split(" ")
    # return ' '.join(c for c  in l1 if len(c)>=3)
    return ' '.join(l1)
def ReadJson(file):
    listdict = []
    fobject = open(file,encoding="utf8")
    for i in fobject.readlines():
        listdict.append(json.loads(i))
    return listdict
if __name__ == "__main__":
    unparsed = []
    temp = Parallel(n_jobs = -1,prefer = "threads")(delayed(ReadJson)(file) for file in files)
    # print(temp)
    for i in temp:
        for j in i:
            # unparsed.append(i.replace("\n",""))
            # print(j)
            if j["lang"]=="en":
                unparsed.append(j["text"].replace("\n",""))
                tdm.add_doc(clean(j["text"].replace("\n","")))
# Lda = gensim.models.ldamodel.LdaModel
    # print(len(unparsed))
# for row in tdm.rows(cutoff=1):
#     print(row)
tdm1 = [x for x in tdm.rows(cutoff=1)]
vocab = tdm1[0]
model = lda.LDA(n_topics=1, n_iter=1000, random_state=1)
td = tdm1[1:]
td = np.array(td).reshape(np.alen(td[0]),np.alen(td))
# print(np.alen(td))
# td = td.reshape()
model.fit(td)
topic_word = model.topic_word_
n_top_words = 7

for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
# print(tdm1[1:])
# ldamodel = Lda(tdm1[1:], num_topics=1, id2word = None, passes=50)
# print(ldamodel.print_topics(num_topics=1, num_words=3))


#NMF

model = NMF(n_components=1, init='random', random_state=0).fit(td)
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 8
display_topics(model, vocab, no_top_words)
