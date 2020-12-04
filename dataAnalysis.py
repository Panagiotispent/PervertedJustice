# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 10:44:44 2020

@author: panay
"""
import pandas as pd
import sqlite3
import regex as re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob



#create dataframe from csv
df = pd.read_csv(r'./log.csv')
mask1 = [isinstance(item, (str, bytes)) for item in df['Username']]


df = df.loc[mask1]

mask2 = [isinstance(item, (str, bytes)) for item in df['Time']]
df = df.loc[mask2]
mask3 = [isinstance(item, (str, bytes)) for item in df['Message']]
df = df.loc[mask3]#removes the nan and empty calues from the dataset

print(df.head())
print('shape = ',df.shape)

stop_words = ['ok','so','is','you','your','and', 'the', 'to', 'from', 'or', 'I', 'for', 'do', 'get',
              'not', 'here', 'in', 'im', 'have', 'on', 're', 'new', 'subject','such','it'
              ,'why','a','yo','ya','but','of','I','bye']

wordcloud = WordCloud(width = 1000, height = 1000, background_color = 'black', stopwords = stop_words,max_words=100
                      , min_font_size = 20).generate(str(df['Message']))
#plot the word cloud
fig = plt.figure(figsize = (8,8), facecolor = None)
plt.imshow(wordcloud)
plt.axis('off')

plt.show()

df['polarity'] = df['Message'].map(lambda text: TextBlob(text).sentiment.polarity)
df['Message_len'] = df['Message'].astype(str).apply(len)
df['word_count'] = df['Message'].apply(lambda x: len(str(x).split()))

#check the polarity of some messages by using TextBlob
print('5 random messages with the highest positive sentiment polarity: \n')
cl = df.loc[df.polarity == 1, ['Message']].sample(5).values()
for c in cl:
    print(c[0])
    
print('5 random messages with the most neutral sentiment(zero) polarity: \n')
cl = df.loc[df.polarity == 0, ['Message']].sample(5).values
for c in cl:
    print(c[0])
    
print('5 random messages with the most negative polarity: \n')
cl = df.loc[df.polarity == -1, ['Message']].sample(5).values
for c in cl:
    print(c[0])
    
df['polarity'].plot(
    kind='density',
    title='Sentiment Polarity Distribution')
plt.show()


df['Message_len'].plot(
    kind='hist',
    bins=10,
    title='Message Length Distribution')


plt.show()

df['word_count'].plot(
    kind='hist',
    bins=10,
    title='Message Word Count Distribution')

plt.show()
