import numpy as np
import pandas as pd
import nltk
import re

file_name = 'survey_questions.csv'
dataset = pd.read_csv(file_name)
dataset = dataset['Questions'][27:32]

options = { "LIKERT" : ["Strongly Agree", "Agree", "Neutral", "Disagree", "Strongly Disagree"], "YN" : ["Yes", "No"]}

for question in dataset:
    print("\n" + question)
    if question[-1] != "?":
        print(options["LIKERT"])
        user_respond = input("Enter: ")
    else:
        print(options["YN"])
        user_respond = input("Enter: ")
    
print("THE END")
