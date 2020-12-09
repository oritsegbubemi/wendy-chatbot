###############################################################################1
#IMPORT ALL NEEDED LIBRARIES AND MODULES
import re
import nltk
import random
import sqlite3
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

###############################################################################2
#LOAD SURVEY DATASET
file_name = 'survey_questions.csv'
dataset = pd.read_csv(file_name)
questions = dataset['Questions']
labels = dataset['Options']

###############################################################################3
#NLTK FOR GOOD PATTERNING, TOKENIZATION, LEMMATIZATION
nltk.data.path.append('./nltk_data/')
lemmatizer = WordNetLemmatizer()
def stem_tokens(tokens):
    stemmed = []
    for i in tokens:
        stemmed.append(lemmatizer.lemmatize(i))
    return stemmed
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens)
    return stems

################################################################################4
#CLASSIFICATION OF THE SURVEY QUESTIONS
vectorizer = TfidfVectorizer(tokenizer=tokenize, analyzer='word', lowercase=True, stop_words='english')
X_vector = vectorizer.fit_transform(questions)
clf = SVC(kernel='linear', probability=True)
clf.fit(X_vector,labels)

def classifier(message):
    out_put = [message]
    print("Output String: ", out_put)
    out_put_vector = vectorizer.transform(out_put)
    print("Predict Proba: ", clf.predict_proba(out_put_vector))
    out_put_class = clf.predict(out_put_vector)
    print("Output Class: ", out_put_class)
    return out_put_class[0]

classifier("Do you like your company?")
