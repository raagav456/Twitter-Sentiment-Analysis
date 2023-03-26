# Twitter-Sentiment-Analysis

<p align="center">
  <img  src="https://user-images.githubusercontent.com/78891081/227776493-27d7077f-1381-4361-9246-81433f761814.png" width = "600" height = "300" >
</p>

It is a Natural Language Processing Problem where Sentiment Analysis is done by Classifying the Positive tweets from negative tweets by machine learning models for classification, text mining, text analysis, data analysis and data visualization


# Introduction

Natural Language Processing (NLP) is a hotbed of research in data science these days and one of the most common applications of NLP is sentiment analysis. From opinion polls to creating entire marketing strategies, this domain has completely reshaped the way businesses work, which is why this is an area every data scientist must be familiar with.

Thousands of text documents can be processed for sentiment (and other features including named entities, topics, themes, etc.) in seconds, compared to the hours it would take a team of people to manually complete the same task.

We will do so by following a sequence of steps needed to solve a general sentiment analysis problem. We will start with preprocessing and cleaning of the raw text of the tweets. Then we will explore the cleaned text and try to get some intuition about the context of the tweets. After that, we will extract numerical features from the data and finally use these feature sets to train models and identify the sentiments of the tweets.

This is one of the most interesting challenges in NLP so I’m very excited to take this journey with you!

# Understand the Problem Statement

Let’s go through the problem statement once as it is very crucial to understand the objective before working on the dataset. The problem statement is as follows:

The objective of this task is to detect hate speech in tweets. For the sake of simplicity, we say a tweet contains hate speech if it has a racist or sexist sentiment associated with it. So, the task is to classify racist or sexist tweets from other tweets.

Formally, given a training sample of tweets and labels, where label ‘1’ denotes the tweet is racist/sexist and label ‘0’ denotes the tweet is not racist/sexist, your objective is to predict the labels on the given test dataset.

Note: The evaluation metric from this practice problem is F1-Score.

# Pipeline

<p align="center">
  <img  src="https://user-images.githubusercontent.com/78891081/227777049-e49b9e40-6d29-4d84-a9c2-b0d97a0dbcc0.jpg" width = "600" height = "300" >
</p>

# Twitter PreProcessing and Cleaning

The preprocessing of the text data is an essential step as it makes the raw text ready for mining, i.e., it becomes easier to extract information from the text and apply machine learning algorithms to it. If we skip this step then there is a higher chance that you are working with noisy and inconsistent data. The objective of this step is to clean noise those are less relevant to find the sentiment of tweets such as punctuation, special characters, numbers, and terms which don’t carry much weightage in context to the text.

# Story Generation and Visualzation from tweets

In this section, we will explore the cleaned tweets text. Exploring and visualizing data, no matter whether its text or any other data, is an essential step in gaining insights. Do not limit yourself to only these methods told in this tutorial, feel free to explore the data as much as possible.

Before we begin exploration, we must think and ask questions related to the data in hand. A few probable questions are as follows:

What are the most common words in the entire dataset? What are the most common words in the dataset for negative and positive tweets, respectively? How many hashtags are there in a tweet? Which trends are associated with my dataset? Which trends are associated with either of the sentiments? Are they compatible with the sentiments?

## Frequency of words

<p align="center">
  <img  src="https://user-images.githubusercontent.com/78891081/227777677-21048223-ddd4-44db-9031-01c95e9ef453.png" width = "600" height = "300" >
</p>

## Frequency of Positive words

<p align="center">
  <img  src="https://user-images.githubusercontent.com/78891081/227777702-b6963ed1-4204-4548-be64-e27ec6a24f48.png" width = "600" height = "300" >
</p>

## Frequency of Negative words

<p align="center">
  <img  src="https://user-images.githubusercontent.com/78891081/227777718-0fdece78-19fb-490e-8bff-961c5b16f388.png" width = "600" height = "300" >
</p>

## Confusion Matrix

<p align="center">
  <img  src="https://user-images.githubusercontent.com/78891081/227777974-70a02f72-7d64-47b6-9e54-7754bc9f40c1.png" width = "600" height = "300" >
</p>

# Visualizing the Tweets

![download (4)](https://user-images.githubusercontent.com/78891081/227779215-bb22f6aa-e7fc-4cd6-80e8-a45ce11cf9d2.png)


# Main Packages Used
  
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
```
