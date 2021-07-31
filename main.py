import pandas as pd
from preprocessing import select_tweets
from sentiment_analysis import keywordBased, mlBased



'''

0. (Build model)

1. Load our data

2. Run model on the data


'''

# Load our data
tweets = pd.read_csv('data/trump_add_lang.csv', lineterminator='\n', parse_dates=True)  # (776886, 21)
tweets = select_tweets.select_tweets(tweets)
print("-- Total number of entries:", len(tweets))


# Run model
TB_instance = keywordBased.TB()
VADER_instance = keywordBased.VADER()
ML_instance = mlBased.ML()

tweets['sentiment_TB'] = tweets['tweet_clean'].apply(TB_instance.predict)
tweets['sentiment_VADER'] = tweets['tweet_clean'].apply(VADER_instance.predict)
tweets['sentiment_ML'] = tweets['tweet_clean'].apply(ML_instance.predict)


# Save (opt)
tweets.to_csv('data/with_prediction_trump.csv')





