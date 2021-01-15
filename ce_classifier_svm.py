###############################################################################1
#IMPORT ALL NEEDED LIBRARIES AND MODULES
import nltk
import pickle
import string
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix


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
    stemmed = []
    for i in tokens:
        stemmed.append(lemmatizer.lemmatize(i))
    return stemmed

def tokenize(text):
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    return lem_tokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


################################################################################4
#TRAINING THE MODEL
vectorizer = TfidfVectorizer(tokenizer=tokenize, analyzer='word', lowercase=True)
X_vector = vectorizer.fit_transform(questions)
X_train, X_test, y_train, y_test = train_test_split(X_vector, labels, test_size=0.20, random_state=101)

classifier = SVC(C=100, gamma=0.1, kernel='linear', probability=True)
classifier.fit(X_train, y_train)
filename = 'pickle_model.pkl'
pickle.dump(classifier, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
predictions = loaded_model.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))




###############
# Gridsearch
# param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
# from sklearn.model_selection import GridSearchCV
# grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
# grid.fit(X_train,y_train)
# print("The Best: ", grid.best_estimator_)
# grid_predictions = grid.predict(X_test)
# print(confusion_matrix(y_test,grid_predictions))
# print(classification_report(y_test,grid_predictions))
