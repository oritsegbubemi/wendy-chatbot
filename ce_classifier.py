###############################################################################1
#IMPORT ALL NEEDED LIBRARIES AND MODULES
import re
import nltk
import string
import random
import sqlite3
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

def lem_tokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]


def tokenize(text):
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    return lem_tokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

################################################################################4
#CLASSIFICATION OF THE SURVEY QUESTIONS
vectorizer = TfidfVectorizer(tokenizer=tokenize, analyzer='word', lowercase=True)
X_vector = vectorizer.fit_transform(questions)
clf = SVC(kernel='linear', probability=True)
clf.fit(X_vector, labels)

def classifier(message):
    output = []
    message_tokens = tokenize(message)
    message_string = (' '.join(message_tokens))
    output.append(message_string)
    output_vector = vectorizer.transform(output)
    print(clf.predict_proba(output_vector))
    output_class = clf.predict(output_vector)
    return output_class

print(classifier("Are you doing fine?"))