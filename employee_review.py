from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    print("{} {}".format(sentence, str(score)))
    if (score['compound'] >= 0.67):
        print("Very Happy")
    elif (score['compound'] >= 0.33 and score['compound'] < 0.67):
        print("Happy")
    elif (score['compound'] >= -0.33 and score['compound'] < 0.33):
        print("Neutral")
    elif (score['compound'] >= -0.67 and score['compound'] < -0.33):
        print("Quite Unhappy")
    else:
        print("Unhappy")

sentiment_analyzer_scores("I am very SAD!!")