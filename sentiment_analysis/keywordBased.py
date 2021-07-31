from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from sentiment_analysis.mlBased import preprocess_tweet



'''
TODO: Uniform the output

'''


class TB:

    def __init__(self):
        self.cnt = 0


    def predict(self, tweet):
        print('-- TB running:', self.cnt, end='\r')
        self.cnt += 1
        return TextBlob(preprocess_tweet(tweet)).sentiment.polarity



class VADER:

    def __init__(self):
        self.cnt = 0
        self.analyzer = SentimentIntensityAnalyzer()

    def predict(self, tweet):
        print('-- VADER running:', self.cnt, end='\r')
        self.cnt += 1
        return self.analyzer.polarity_scores(preprocess_tweet(tweet))['compound']



if __name__ == '__main__':

    test_tweet = 'Y’all Just Locking Up Accounts Cause YALL Dont Want The Truth To Come Out About #Biden That’s Crazy Af https://t.co/uNvYFeVTU9'
