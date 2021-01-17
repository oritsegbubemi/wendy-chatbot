import pickle
from model_svm import survey_question


# with open('vectorizer_model.pkl', 'rb') as file:  
#     vectorizer = pickle.load(file)

# with open('classifier_model.pkl', 'rb') as file:  
#     classifier = pickle.load(file)


# def survey_question(question):
#     test_question_transform = vectorizer.transform(question)
#     result = classifier.predict(test_question_transform)
#     option = {"LIKERT" : ["Strongly Agree", "Agree", "Neutral", "Disagree", "Strongly Disagree"], "YN" : ["Yes", "No"]}
#     return option[result[0]]

print(survey_question(["Do you understand you daily tasks?"]))