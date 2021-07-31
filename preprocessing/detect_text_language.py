import pandas as pd
from fast_lang_detect import detect

tweets_b = pd.read_csv('/Users/yujia/Desktop/NWU/510_Social_Media/Proj/archive/hashtag_joebiden.csv', lineterminator='\n', parse_dates=True)  # (776886, 21)
tweets_t = pd.read_csv('/Users/yujia/Desktop/NWU/510_Social_Media/Proj/archive/hashtag_donaldtrump.csv', lineterminator='\n', parse_dates=True)  # (970919, 21)

def custom_detect(x):
  try:
    return detect(x)
  except:
    return "Link"

print('Start')

tweets_b["tweet_lang"] = tweets_b["tweet"].apply(custom_detect)
print("Biden tweets: Finished adding language")
tweets_t["tweet_lang"] = tweets_t["tweet"].apply(custom_detect)
print("Trump tweets: Finished adding language")

tweets_b.to_csv('/Users/yujia/Desktop/NWU/510_Social_Media/Proj/archive/biden_add_lang.csv', index = False)
tweets_t.to_csv('/Users/yujia/Desktop/NWU/510_Social_Media/Proj/archive/trump_add_lang.csv', index = False)

