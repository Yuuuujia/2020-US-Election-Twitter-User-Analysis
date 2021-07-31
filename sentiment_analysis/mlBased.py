import numpy as np
import pandas as pd
import re, pickle, os


def import_tweets(filename, header=None):
    # import data from csv file via pandas library
    tweet_dataset = pd.read_csv(filename, encoding='latin', header=header)
    # the column names are based on sentiment140 dataset provided on kaggle
    tweet_dataset.columns = ['sentiment', 'id', 'date', 'flag', 'user', 'text']
    # delete 3 columns: flags,id,user, as they are not required for analysis
    tweet_dataset = tweet_dataset.drop(["id","user","date","user"], axis = 1)
    # in sentiment140 dataset, positive = 4, negative = 0; So we change positive to 1
    tweet_dataset.sentiment = tweet_dataset.sentiment.replace(4, 1)
    return tweet_dataset



def preprocess_tweet(tweet):
    # Preprocess the text in a single tweet
    # arguments: tweet = a single tweet in form of string
    # convert the tweet to lower case
    tweet.lower()
    # convert all urls to sting "URL"
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)
    # convert all @username to "AT_USER"
    tweet = re.sub('@[^\s]+', 'AT_USER', tweet)
    # correct all multiple white spaces to a single white space
    tweet = re.sub('[\s]+', ' ', tweet)
    # convert "#topic" to just "topic"
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

    tweet = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split()

    return ' '.join(tweet)



def feature_extraction(data, tfv=None):

    # TODO: Other vectorization methods available

    if(not tfv):
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfv = TfidfVectorizer(sublinear_tf=True, stop_words="english")  # we need to give proper stopwords list for better performance
        features = tfv.fit_transform(data)

    else:
        features = tfv.transform(data)

    return features, tfv



def train():

    tweet_dataset = import_tweets("data/sentiment_140.csv")
    tweet_dataset['text'] = tweet_dataset['text'].apply(preprocess_tweet)
    data = np.array(tweet_dataset.text)
    label = np.array(tweet_dataset.sentiment)
    features, tfv = feature_extraction(data)

    print("-- Training data loaded and processed")

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=1., max_iter=1000, verbose=1)
    model.fit(features, label)

    return model, tfv




def finetune_model():
    tweet_dataset = import_tweets("data/sentiment_140.csv")
    tweet_dataset['text'] = tweet_dataset['text'].apply(preprocess_tweet)
    data = np.array(tweet_dataset.text)
    label = np.array(tweet_dataset.sentiment)
    features, tfv = feature_extraction(data)

    print("-- Training data loaded and processed")

    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import BernoulliNB, ComplementNB
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score
    from sklearn.svm import SVC, LinearSVC, NuSVC
    import lightgbm
    import catboost
    import xgboost


    kf = StratifiedKFold(n_splits=5)

    for train_index, test_index in kf.split(features, label):
        x_train, y_train = features[train_index], label[train_index]
        x_test, y_test = features[test_index], label[test_index]


        '''
        Available models:
        Sklearn:
            Logistic regression # C=1., max_iter=1000 -> 77%
            Naive Bayes # BernoulliNB -> 75% ~ 76% (fast) || ComplementNB -> 75% ~ 76%
            SVM # LinearSVC -> 76%
            Neural Networks # 
            Tree-based # 
            Others # 
            
        Lightgbm # Lightgbm (binary, 1000) -> 77%
        XGBoost # Default -> 72% ~ 73%
        
        '''

        tmp_model = LogisticRegression(C=1.0, max_iter=1000, verbose=1)

        print("-- Training starts")
        tmp_model.fit(x_train, y_train)

        y_pred = tmp_model.predict(x_test)

        print("-- Training complete:", accuracy_score(y_test, y_pred))





class ML:

    def __init__(self):
        self.cnt = 0
        self.model, self.tfv = train()


    def predict(self, tweet):
        print('-- ML running:', self.cnt, end='\r')
        self.cnt += 1
        cleaned_tweet = np.array([preprocess_tweet(tweet)])
        converted_tweet, _ = feature_extraction(cleaned_tweet, self.tfv)
        result = self.model.predict_proba(converted_tweet)

        return (0.5 - result[0][0]) * 2




if __name__ == '__main__':

    # test_tweet = 'Y’all Just Locking Up Accounts Cause YALL Dont Want The Truth To Come Out About #Biden That’s Crazy Af https://t.co/uNvYFeVTU9'
    #
    # ML_instance = ML()
    #
    # print(ML_instance.predict(test_tweet))

    finetune_model()

