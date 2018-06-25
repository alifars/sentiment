# coding: utf-8
import string
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from keras.models import load_model
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
import re

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
test_texts = test_df["Tweet"]
train_texts = train_df["tokens"]

labels = train_df["class_label"]
print(labels.shape)
encoded_labels = pd.get_dummies(labels)

encoded_labels.drop(encoded_labels.columns[2], axis=1, inplace=True)

def clean_text(text):
    translator = str.maketrans("", "", string.punctuation)
    return  text.translate(translator)

def vectorize_text(texts):
    texts = [clean_text(text) for text in texts]
    tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(),
                                       ngram_range=(1, 3), max_features=18000, lowercase=True)
    vectorized_texts = tfidf_vectorizer.fit_transform(texts)
    return vectorized_texts


def ann_model():
    model = Sequential()
    model.add(Dense(100, input_shape=(19000,), activation="relu"))
    Dropout(0.5)
    model.add(Dense(100, activation="relu"))
    Dropout(0.5)
    model.add(Dense(2, activation="sigmoid"))
    print(model.summary())

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
    return model


def randomforest():
    clf = RandomForestClassifier()
    return clf


train_vectors = vectorize_text(train_texts)
test_vectors = vectorize_text(test_texts)

print(train_vectors.shape)
print(test_vectors.shape)


def train_ann():
    model = ann_model()
    model.fit(train_vectors, encoded_labels, epochs=10, batch_size=10, verbose=1)
    model.save("model.h5")


def logistic_pipeline():
    lr = LogisticRegression(C=1)
    lr.fit(train_vectors, labels)



    # preds = lr.predict(test_vectors)
    # write_predictions(preds)
def naive_bayes_pipeline():
    nb = BernoulliNB()
    nb.fit(train_vectors, labels)

    preds = nb.predict(test_vectors)
    write_predictions(preds)


def svm_pipeline():
    clf = SVC(C=1, gamma=1, kernel="rbf")
    clf.fit(train_vectors, labels)
    preds = clf.predict(test_vectors)
    write_predictions(preds)



def xgb_pipline():
    clf = XGBClassifier()
    clf.fit(train_vectors, labels)
    preds = clf.predict(test_vectors)
    for i in preds:
        print(i)


def write_predictions(predictions):
    with open("subm.csv", "w") as f:
        f.write("Id" + "," + "Expected" + "\n")
        for i, pred in enumerate(predictions):
            f.write(str(i) + "," + str(pred) + "\n")


def pred_ann():
    mymodel = load_model("model.h5")
    predictions = mymodel.predict(test_vectors)
    with open("subm.csv", "w") as f:
        f.write("Id" + "," + "Expected" + "\n")
        for i, pred in enumerate(predictions):

            if pred[0] > pred[1]:
                label = 0
            else:
                label = 1

            f.write(str(i) + "," + str(label) + "\n")
            # f.write(str(i) + "," + str(pred) + "\n")


def ann_pipeline():
    train_ann()
    pred_ann()


# ann_pipeline()
# xgb_pipline()
#logistic_pipeline()
naive_bayes_pipeline()
