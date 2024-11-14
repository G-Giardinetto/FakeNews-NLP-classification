import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sn
import datasetDownloader as dD

path="WELFake_Dataset.csv"

if not os.path.exists(path):
    print("File does not exist. Downloading")
    dD.download()

df = pd.read_csv(path)
print("Dataframe before drop")
df.head()
df.drop('Unnamed: 0', axis='columns', inplace=True)
print("Dataframe after drop")
df.head()
sn.barplot(x='label', data = df, palette= 'mako')
plt.show()