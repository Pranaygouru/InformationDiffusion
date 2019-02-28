import os
import glob
import json
import re
List_Hashtags = {}
f = open('FinalTrump','r',encoding="utf8")
for i in f.readlines():
      Dicta = json.loads(i)
      if "extended_tweet" in Dicta.keys():
          if "#" in Dicta["extended_tweet"]["full_text"] and Dicta["lang"] == "en":
            line = Dicta["extended_tweet"]["full_text"].replace("\n","")
            line = line.replace("https:"," ")
            arra = line.split('#')
            for t in range(1,len(arra)):
                val = arra[t].split(" ")
                arra[t] = val[0]
                if arra[t] in List_Hashtags.keys():
                    val = List_Hashtags.get(arra[t])
                    val1 = val + 1
                    List_Hashtags[arra[t]] = val1
                else:
                    List_Hashtags[arra[t]] = 1
            # if matchobj:
            #     print(matchobj.group())
      elif "#" in Dicta["text"] and Dicta["lang"] == "en":
           line = Dicta["text"].replace("\n","")
           line = line.replace("https:"," ")
           # matchobj = re.match((".*#[a-z][ ]"), line, re.M | re.I)
           arra = line.split('#')
           for t in range(1, len(arra)):
               val = arra[t].split(" ")
               arra[t] = val[0]
               if arra[t] in List_Hashtags.keys():
                   val = List_Hashtags.get(arra[t])
                   val1 = val + 1
                   List_Hashtags[arra[t]] = val1
               else:
                   List_Hashtags[arra[t]] = 1
d = {}
d = {(k,v) for k,v in List_Hashtags.items() if v >= 20}
print(d)





