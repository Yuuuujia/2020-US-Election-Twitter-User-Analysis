

import pandas as pd
import matplotlib.pyplot as plt
from time import time

'''
tweet_clean, tweet_lang, likes, retweet_count, source
user_join_date,user_followers_count,user_location,lat,long,city,country,continent,state
sentiment_ML


#Trump / #DonaldTrump
#Biden / #JoeBiden
'''



def load_data(name='Trump'):
    if(name == 'Trump'): file_path = 'data/with_prediction_trump.csv'
    else: file_path = 'data/with_prediction_biden.csv'


    return pd.read_csv(file_path, lineterminator='\n', parse_dates=True)




def get_pie_chart():

    biden_sizes = [66.8, 7.4, 5.4, 4.7, 4.3, 11.4]
    biden_labels = ['English', 'Spanish', 'Dutch', 'German', 'French', 'Others']

    trump_sizes = [70.3, 6.2, 4.5, 4.5, 2.9, 11.6]
    trump_labels = ['English', 'Spanish', 'German', 'French', 'Italian', 'Others']

    plt.pie(trump_sizes, labels=trump_labels, autopct='%1.1f%%')
    plt.title("Tweets tagged #Trump or #DonaldTrump")
    plt.show()




def get_distribution(data, label):

    plt.hist(data[label], bins=100)
    plt.title('Predicted sentiments for tweet containing #Trump / #DonaldTrump')
    plt.show()



if __name__ == "__main__":

    start = time()
    data = load_data('Trump')
    print("-- Data loaded! It takes", round(time() - start, 3), 'seconds to load the dataset')


    get_distribution(data, 'sentiment_ML')



