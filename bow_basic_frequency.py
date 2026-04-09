from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import re

data=pd.read_csv('music.txt',sep="\t",names=["caption"])

corpus=[]
for i in range(len(data)):
    statement=re.sub("[^a-zA-Z]"," ",data["caption"][i])
    statement=statement.lower()
    statement=statement.split()
    statement=[WordNetLemmatizer().lemmatize(word) for word in statement
               if word not in set(stopwords.words('english'))]
    statement=" ".join(statement)
    corpus.append(statement)

cv1=CountVectorizer(max_features=5,binary=True)
matrix1=cv1.fit_transform(corpus)
features_top1=cv1.get_feature_names_out()
key_freq1={key:int(count) for key,count in zip(features_top1,matrix1.sum(axis=0).A1)}
sorted_key_freq1= sorted(key_freq1.items(), key=lambda x: x[1], reverse=True)

print(features_top1)
print(key_freq1)
print(sorted_key_freq1)

"""
Here key:music is used twice in a statement, while it's counted as '1', as 'binary=TRUE',
it dosen't count the frequency, but returns '1' if the key is present in the statement, 
irrespective of multiple times.

So we can remove 'binary=TRUE' to count the frequency.
"""

cv2=CountVectorizer(max_features=5)
matrix2=cv2.fit_transform(corpus)
features_top2=cv2.get_feature_names_out()
key_freq2={key:int(count) for key,count in zip(features_top2,matrix2.sum(axis=0).A1)}
sorted_key_freq2=sorted(key_freq2.items(), key=lambda x: x[1], reverse=True)

print(features_top2)
print(key_freq2)
print(sorted_key_freq2)