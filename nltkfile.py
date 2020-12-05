import pandas as pd
import numpy as np
import nltk
import re

file_name = 'survey_questions.csv'
dataset = pd.read_csv(file_name)
dataset = dataset['Survey Questions']

first_question = dataset[0]
print(first_question)
token = nltk.word_tokenize(first_question)
print(token)
tagged = nltk.pos_tag(token)
print(tagged)

