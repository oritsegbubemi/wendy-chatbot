
import os
import pickle
from flask import Flask, jsonify, request
from flask_restful import Resource, Api
from model import tokenize


app = Flask(__name__)
api = Api(app)

with open('model/vectorizer_model.pkl', 'rb') as file:  
    vectorizer = pickle.load(file)

with open('model/classifier_model.pkl', 'rb') as file:  
    classifier = pickle.load(file)


class MakePrediction(Resource):
    @staticmethod
    def post():
        posted_data = request.get_json()
        print(posted_data)
        survey_question = posted_data['survey_question']
        
        survey_question = vectorizer.transform([survey_question])
        print(survey_question)

        prediction = classifier.predict(survey_question)
        if prediction == 'LIKERT':
            predicted_class = ["Strongly Agree", "Agree", "Neutral", "Disagree", "Strongly Disagree"]
        elif prediction == 'YN':
            predicted_class = ["Yes", "No"]
        return jsonify({
            'Prediction': predicted_class
        })


api.add_resource(MakePrediction, '/predict')


if __name__ == '__main__':
    app.run(debug=True)