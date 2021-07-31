import pandas as pd
import re


def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


def select_tweets(df):

    print("Select English tweets")
    df_en = df[df.tweet_lang == 'en']

    print("Select tweets with location and source")
    df_en_loc = df_en[((df_en.user_location.isnull() == False)
                      | (df_en.city.isnull() == False)
                      | ((df_en.lat.isnull() == False) & (df_en.long.isnull() == False))
                      | (df_en.state.isnull() == False)
                      | (df_en.state_code.isnull() == False))
                      & (df_en.source.isnull() == False)]

    print("Clean tweets")
    df_en_loc["tweet_clean"] = df_en_loc["tweet"].apply(clean_tweet)

    print("Select tweets with length lager than 10")
    df_out = df_en_loc[df_en_loc.tweet_clean.apply(lambda x: len(str(x)) > 10)]

    print("Finished!")

    return df_out
