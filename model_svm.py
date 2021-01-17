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

# saving model to a pickel file
filename = 'pickle_model.pkl'
with open(filename, 'wb') as file:  
    pickle.dump(classifier, file)

filename1 = 'vectorizer_model.pkl'
with open(filename1, 'wb') as file:  
    pickle.dump(vectorizer, file)
################################################################################4
#TESTING THE MODEL
with open(filename, 'rb') as file:  
    loaded_model = pickle.load(file)

with open(filename1, 'rb') as file:  
    loaded_vectorize = pickle.load(file)
    
# score = loaded_model.score(X_test, y_test)  
# predictions = loaded_model.predict(X_test)
#print(score)
#print(predictions)
#print(confusion_matrix(y_test,predictions))
#print(classification_report(y_test,predictions))


################################################################################4
#USING THE MODEL THE MODEL
def survey_question(question):
    test_question_transform = loaded_vectorize.transform(question)
    result = loaded_model.predict(test_question_transform)
    option = {"LIKERT" : ["Strongly Agree", "Agree", "Neutral", "Disagree", "Strongly Disagree"], "YN" : ["Yes", "No"]}
    return option[result[0]]