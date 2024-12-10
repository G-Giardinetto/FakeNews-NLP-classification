import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import datasetDownloader as dD
import nltk
import torch
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from time import time
from transformers import AutoModel, AutoTokenizer
from torch.amp import autocast
import gc

torch.cuda.empty_cache()
gc.collect()


start = time()
DROP_ROWS=50000
maxIter=6000

path="WELFake_Dataset.csv"
choice=None
model_name = 'albert-base-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
modello = AutoModel.from_pretrained(model_name)
modello.eval()
if torch.cuda.is_available():
    print("CUDA available")
else:
    print("CUDA not available")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))

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

def get_bert_embeddings(sentences):
    modello.to(device)
    embeddings_list = []
    for sentenze in sentences:
        encoded_inputs = tokenizer(sentenze, padding='max_length', truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            with autocast(device_type='cuda'):
                outputs = modello(**encoded_inputs, output_hidden_states=True )
                embeddings = outputs.pooler_output
                embeddings_list.append(embeddings.cpu().squeeze(0).tolist())
    return embeddings_list





preprotitles=os.path.exists("PreprocessedTitlesDELETED.csv")
preprotexts=os.path.exists("PreprocessedTextsDELETED.csv")
#######Data preprocessing
if preprotexts or preprotexts:
    if preprotitles:
        print("Already exists preprocessed data. Loading texts...")
        df = pd.read_csv("PreprocessedTitlesDELETED.csv")
    else:
        print("Already exists preprocessed data. Loading titles...")
        df = pd.read_csv("PreprocessedTextsDELETED.csv")
else:
    print("Preprocessed data does not exist yet.")
    if not os.path.exists(path):
        print("File does not exist. Downloading")
        dD.download()
    else:
        print("File already downloaded")

    df = pd.read_csv(path)
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


    ####### Stemming
    print("\nDo you want to delete the titles or texts? (1/2)")
    if int(input()) == 2:
        print("Deleting texts")
        df['title'] = df['title'].apply(stemm)
        df.drop(columns='text', inplace=True)
        choice="TextsDELETED"
        print("\nDataframe's stemmed titles:")
        print(df['title'].head(3))
    else:
        print("Deleting titles")
        df['text'] = df['text'].apply(stemm)
        df.drop(columns='title', inplace=True)
        choice="TitlesDELETED"
        print("\nDataframe's stemmed texts:")
        print(df['texts'].head(3))

    df.to_csv("Preprocessed"+choice+".csv", index=False)

### Plot data
sn.countplot(x='label', hue='label',legend=False, data = df, palette= 'mako')
plt.show()

## Data engineering
if choice is not None:
    if choice== "TitlesDELETED":
        X=df['texts']
        print("Using texts")
    else:
        X=df['title']
        print("Using titles")
else:
    if preprotexts:
        X=df['title']
        print("Using titles")
    else:
        X=df['texts']
        print("Using texts")

Y=df['label'].values
print("Preprocessed data:\n")
print(X.head(3)+"\n")


### Embedding
print("Generating embeddings using {}...".format(model_name))
X = get_bert_embeddings(X)

#Trimming
if len(X) != len(Y):
    min_length = min(len(X), len(Y))
    X= X[:min_length]
    Y = Y[:min_length]

### SPLIT DATAFRAME
print("Random splitting data into training and testing...")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

### REGRESSION
lr = LogisticRegression()
lr.max_iter = maxIter
print("Model name: Logistic Regression")
print("Training with {} iterations...".format(maxIter))

lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_test)



### EVALUATING ACCURACY
accuracy = accuracy_score(Y_pred, Y_test)

print(classification_report(Y_test,Y_pred)+"\n")

### CONFUSION MATRIX
sn.heatmap(confusion_matrix(Y_test,Y_pred),annot = True, cmap = 'Greens',fmt = '.1f')
plt.show()


X_new_test="The president of the United States just declared war to Italy for enslaving the population of Iran"
print("New test: {}".format(X_new_test))
X_new_test = get_bert_embeddings([X_new_test])
Y_new_pred = lr.predict(X_new_test)
print(str(Y_new_pred)+"\t0: Fake, 1: True")

end = time()
print("Elapsed time {}".format(end-start))