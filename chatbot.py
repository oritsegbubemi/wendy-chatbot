import pandas as pd
import numpy as np
import nltk
import re

file_name = 'survey_questions.csv'
dataset = pd.read_csv(file_name)
dataset = dataset['Survey Questions'][32:35]

options = { "first" : ["Strongly Agree", "Agree", "Neutral", "Disagree", "Strongly Disagree"], "second" : ["Yes", "No", "Not Sure"]}

for question in dataset:
    if question[-1] == "?":
        print(question)
        print(options["second"])
        user_respond = input("Enter: ")
    else:
        print(question)
        print(options["first"])
        user_respond = input("Enter: ")
