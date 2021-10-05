# Importing modules and printing  Dataset 

import pandas as pd 
import numpy as np 


import re
import string

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
nltk.download('vader_lexicon')


from collections import Counter

from matplotlib import pyplot as plt
from matplotlib import ticker
import seaborn as sns
import plotly.express as px

sns.set(style="darkgrid")

df = pd.read_csv('https://raw.githubusercontent.com/gabrielpreda/covid-19-tweets/master/covid19_tweets.csv')
df.head()

# Selecting only required columns that are needed for the analysis

required_col = ['user_name','date','text']
df = df[required_col]
df.head()

# Convert data type of certain required coloumns for ease of analysis

df.user_name = df.user_name.astype('category')
df.user_name = df.user_name.cat.codes

df.date = pd.to_datetime(df.date).dt.date
df.head()

# Selecting tweet texts for data preprocessing 

tweet_text = df['text']
tweet_text

# Simplifying data by removing URLs

url_removed_tweets = lambda x: re.sub(r'https\S+' , '', str(x))
texts_1 = tweet_text.apply(url_removed_tweets)
texts_1

# Making all tweets lowercase

lowercase_tweets = lambda x: x.lower()
texts_1_2 = texts_1.apply(lowercase_tweets)
texts_1_2 

# Cutting out unnecessary punctuations

punctuation_removed_tweets = lambda x: x.translate(str.maketrans('','',string.punctuation))
texts_1_2_3 = texts_1_2.apply(punctuation_removed_tweets)
texts_1_2_3

# Removing out stopwords (Stop words are any word in a stop list which are filtered out before or after processing of natural language data). 
# Stop words can be customized or imported(standard set of stopwords)

stop_words = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than','covid','#coronavirus', '#coronavirusoutbreak', '#coronavirusPandemic', '#covid19', '#covid_19', '#epitwitter', '#ihavecorona', 'amp', 'coronavirus', 'covid19']

stopwords_removed_tweets = lambda x: ' '.join([word for word in x.split() if word not in stop_words])
texts_1_2_3_4 = texts_1_2_3.apply(stopwords_removed_tweets)
texts_1_2_3_4

# Creating list of words out of tweets and printing a bar graph of most common to least common words

words_list = [word for line in texts_1_2_3_4 for word in line.split()]
words_list[:5]

count1 = Counter(words_list).most_common(50)
words_df = pd.DataFrame(count1)
words_df.columns = ['word','frq']

px.bar(words_df, x='word',y='frq',title = 'Most Common Words')

# Adding the modified text into a main dataframe

df.text = texts_1_2_3_4
df.head()

# Finding polarity scores for each tweet

sid = SentimentIntensityAnalyzer()
score_polarity = lambda x : sid.polarity_scores(x)
score_sentiment = df.text.apply(score_polarity)
score_sentiment

sentiment_df = pd.DataFrame(data = list(score_sentiment))
sentiment_df.head()

# Labelling scores according to the compound polarity score

labelling_score = lambda x : 'neutral' if x == 0 else ('positive' if x>0 else 'negative')
sentiment_df['label'] = sentiment_df.compound.apply(labelling_score)
sentiment_df.head()

# Combining the two dataframes

data = df.join(sentiment_df.label)
data.head()

# Plotting sentiment score counts of all labels

count_df = data.label.value_counts().reset_index()
count_df

sns.barplot(x = 'index', y = 'label', data = count_df)

data1 = data[['user_name','date','label']].groupby(['date','label']).count().reset_index()
data1.columns = ['date','label','counts']
data1.head()

px.line(data1, x = 'date', y = 'counts', color = 'label', title = 'Sentiment Analysis of COVID-19 Tweets')
