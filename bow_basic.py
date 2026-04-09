from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import re

data=pd.read_csv("music.txt",sep="\t",names=["caption"])
# print(data)

corpus=[]
for i in range(len(data)):
    statement=re.sub("[^a-zA-Z]"," ",data["caption"][i])
    statement=statement.lower()
    statement=statement.split()
    statement=[WordNetLemmatizer().lemmatize(word) for word in statement
               if word not in stopwords.words('english')]
    statement=" ".join(statement)
    corpus.append(statement)

# print(corpus)

cv=CountVectorizer(max_features=5,binary=True)
matrix=cv.fit_transform(corpus)
array=matrix.toarray()
features_top=cv.get_feature_names_out()
features={word:int(count) for word,count in cv.vocabulary_.items()}

print(array)
print(features_top)
print(features)


"""
- The matrix array is always represented by the words/features alphabetically.
- 'features_top': Shows all the most frequent features/words in alohabetical manner.
- 'features': Shows all the most frequent features/words based on first occurance in the
text/doc along with their index from the matrix array.
"""
