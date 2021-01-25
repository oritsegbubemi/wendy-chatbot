import pickle
from model import tokenize

classifier_file = 'model/classifier_model.pkl'
with open(classifier_file, 'rb') as file:  
    loaded_classifier = pickle.load(file)

vectorizer_file = 'model/vectorizer_model.pkl'
with open(vectorizer_file, 'rb') as file:  
    loaded_vectorize = pickle.load(file)

def survey_question(question):
    test_question_transform = loaded_vectorize.transform(question)
    result = loaded_classifier.predict(test_question_transform)
    option = {"LIKERT" : ["Strongly Agree", "Agree", "Neutral", "Disagree", "Strongly Disagree"], "YN" : ["Yes", "No"]}
    return option[result[0]]

print(survey_question(["I feel valued at my company"]))