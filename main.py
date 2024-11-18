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
from transformers import AutoModelForSequenceClassification, AutoTokenizer



start = time()
DROP_ROWS=65000

path="WELFake_Dataset.csv"

model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
modello = AutoModelForSequenceClassification.from_pretrained(model_name)  # For  
modello.eval()
if torch.cuda.is_available():
    print("CUDA available")
else:
    print("CUDA not available")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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

def create_windows(dataFrame, window_size, overlap):

    finestre = []
    wstart = 0
    wend = window_size
    while wend <= len(dataFrame):
        window = dataFrame[wstart:wend]
        finestre.append(window)
        wstart += (window_size - overlap)
        wend = wstart + window_size
    return finestre

def get_bert_embeddings(sentences):
    encoded_inputs = tokenizer(sentences, padding='max_length', truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = modello(**encoded_inputs)
        embeddings = outputs.pooler_output
    return embeddings.cpu().tolist()






#######Data preprocessing
if os.path.exists("Preprocessed.csv"):
    print("Already exists preprocessed data. Loading.")
    df = pd.read_csv("Preprocessed.csv")
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
    if input() == '2':
        df['title'] = df['title'].apply(stemm)
        df.drop(columns='text', inplace=True)
        choice="TextsDELETED"
    else:
        df['text'] = df['text'].apply(stemm)
        df.drop(columns='title', inplace=True)
        choice="TitlesDELETED"

    print("\nDataframe's stemmed titles:")
    print(df['title'].head(3))
    df.to_csv("Preprocessed"+choice+".csv", index=False)

### Plot data
sn.countplot(x='label', hue='label',legend=False, data = df, palette= 'mako')
plt.show()

## Data engineering
X=df['text']
Y=df['label'].values

### Embedding
all_windows = []
for sentence in X:
    windows = create_windows(sentence.split(), 512, 256)
    all_windows.extend(windows)
all_windows_str = [' '.join(window) for window in all_windows]

X = get_bert_embeddings(all_windows_str)

print("\nDataframe's embeddings titles:")


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