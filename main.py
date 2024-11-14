import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk import SnowballStemmer
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sn
import datasetDownloader as dD


nltk.download('stopwords')
stemmer = SnowballStemmer("english")

def stem(string):
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
print(df.head())
df.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
print("\nDataframe after drop")
print(df.head())

print("Dataframe rows: {}".format(df.shape[0]))
print("\nDataframe's null elements:")
print(df.isnull().sum())

df.dropna(inplace=True)
print("Dataframe rows: {}".format(df.shape[0]))
print("\nDataframe's null values:")
print(df.isnull().sum())

### Plot results
sn.countplot(x='label', hue='label',legend=False, data = df, palette= 'mako')
plt.show()

####### Stemming
df['text'] = df['text'].apply(stem)
df['title'] = df['title'].apply(stem)

print("\nDataframe's stemmed titles:\n"+df['title'].head())
print("\nDataframe's stemmed texts:\n"+df['text'].head())
