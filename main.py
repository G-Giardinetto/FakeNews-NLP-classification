import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sn
import datasetDownloader as dD

path="WELFake_Dataset.csv"
pd.set_option('display.max_columns', 5)

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

sn.countplot(x='label', hue='label',legend=False, data = df, palette= 'mako')
plt.show()
