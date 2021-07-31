

'''
1. Preprocessing data for ML purposes
2. Build ML models (load existing DL models)
3. Write model.predict(tweet) for unseen data / tweets


'''

import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from tensorflow.keras.preprocessing.text import Tokenizer



def load_ml_data():


    def label_decoder(label):
        return lab_to_sentiment[label]

    def preprocess(text, stop_words, stemmer, text_cleaning_re, stem=False):
        text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
        tokens = []
        for token in text.split():
            if token not in stop_words:
                if stem:
                    tokens.append(stemmer.stem(token))
                else:
                    tokens.append(token)
        return " ".join(tokens)


    df = pd.read_csv('../data/sentiment_train_labeled.csv', encoding = 'latin',header=None)
    df.columns = ['sentiment', 'id', 'date', 'query', 'user_id', 'text']
    df = df.drop(['id', 'date', 'query', 'user_id'], axis=1)

    lab_to_sentiment = {0: "Negative", 4: "Positive"}

    df.sentiment = df.sentiment.apply(lambda x: label_decoder(x))
    df.head()

    stop_words = stopwords.words('english')
    stemmer = SnowballStemmer('english')
    text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

    df.text = df.text.apply(lambda x: preprocess(x, stop_words, stemmer, text_cleaning_re))

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df.text)


    print(df['sentiment'].value_counts())





if __name__ == '__main__':
    load_ml_data()





