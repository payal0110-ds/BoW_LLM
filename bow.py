from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import re

messages=pd.read_csv("SMSSpamCollection",sep="\t",names=["label","message"])
# print(messages)

corpus=[]
for i in range(len(messages)):
    review=re.sub("[^a-zA-Z]"," ",messages["message"][i])
    review=review.lower()
    review=review.split()
    review=[WordNetLemmatizer().lemmatize(word) for word in review
            if word not in set(stopwords.words('english'))]
    review=" ".join(review)
    corpus.append(review)

# Count frequency of top keys
cv=CountVectorizer(max_features=100)
matrix=cv.fit_transform(corpus)
features_top=cv.get_feature_names_out()
features={key:int(index) for key,index in cv.vocabulary_.items() }
# print(features)
key_freq={key:int(count) for key,count in zip(features_top,matrix.sum(axis=0).A1)}
sorted_key_freq=sorted(key_freq.items(), key=lambda x: x[1], reverse=True)

print(key_freq)
print(sorted_key_freq)

# Export the result in a CSV file
df=pd.DataFrame(sorted_key_freq,columns=["key/word","frequency"])
df.to_csv("word_frequencies.csv", index=False)