import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import datasetDownloader as dD
import nltk
from nltk import SnowballStemmer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from time import time
DROP_ROWS=26000

start = time()
nltk.download('stopwords')
stemmer = SnowballStemmer("english")

def stemm(string):
    string = str(string)
    stemmed_string = string
    stemmed_string = stemmed_string.lower()
    stemmed_string = stemmed_string.split()
    stemmed_string = [stemmer.stem(word) for word in stemmed_string if word not in stopwords.words('english')]
    stemmed_string = ' '.join(stemmed_string)
    return stemmed_string



path="WELFake_Dataset.csv"

if not os.path.exists(path):
    print("File does not exist. Downloading")
    dD.download()
else:
    print("File already downloaded")

df = pd.read_csv(path)

#######Data cleaning
print("Dataframe before drop")
print(df.head(3))
df.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
print("\nDataframe after drop")
print(df.head(3))

print("Dataframe rows: {}".format(df.shape[0]))
print("\nDataframe's null elements:")
print(df.isnull().sum())

df.dropna(inplace=True)
###### DROPPING DROP_ROWS ROWS
df.drop(df.tail(DROP_ROWS).index, inplace=True)

print("Dataframe rows: {}".format(df.shape[0]))
print("\nDataframe's null values:")
print(df.isnull().sum())

### Plot results
sn.countplot(x='label', hue='label',legend=False, data = df, palette= 'mako')
plt.show()

####### Stemming
df['title'] = df['title'].apply(stemm)
df.drop(columns='text', inplace=True)

print("\nDataframe's stemmed titles:")
print(df['title'].head(3))

## Data engineering
X=df['title'].values
Y=df['label'].values

### Embedding

vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X=vectorizer.transform(X)
print("\nDataframe's vectorized titles:")
print(vectorizer.vocabulary_)
print(X)

### SPLIT DATAFRAME
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

### REGRESSION
lr = LogisticRegression()

lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_test)

### EVALUATING ACCURACY
accuracy = accuracy_score(Y_pred, Y_test)

print(classification_report(Y_test,Y_pred))

### CONFUSION MATRIX
sn.heatmap(confusion_matrix(Y_test,Y_pred),annot = True, cmap = 'Greens',fmt = '.1f')
plt.show()


end = time()
print("Elapsed time {}".format(end-start))