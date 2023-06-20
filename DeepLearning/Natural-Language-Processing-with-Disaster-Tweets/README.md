# Natural-Language-Processing-with-Disaster-Tweets

Predict which Tweets are about real disasters and which ones are not


## Introduction

This particular challenge is perfect for data scientists looking to get started with Natural Language Processing. The competition dataset is not too big, and even if you don’t have much personal computing power, you can do all of the work in our free, no-setup, Jupyter Notebooks environment called Kaggle Notebooks.

## Competition Description

Twitter has become an important communication channel in times of emergency.
The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).

But, it’s not always clear whether a person’s words are actually announcing a disaster. Take this example:

<img style="float:left;width:250px;margin-right:10px" src="https://storage.googleapis.com/kaggle-media/competitions/tweet_screenshot.png">

The author explicitly uses the word “ABLAZE” but means it metaphorically. This is clear to a human right away, especially with the visual aid. But it’s less clear to a machine.

In this competition, you’re challenged to build a machine learning model that predicts which Tweets are about real disasters and which one’s aren’t. You’ll have access to a dataset of 10,000 tweets that were hand classified. If this is your first time working on an NLP problem, we've created a quick tutorial to get you up and running.

Disclaimer: The dataset for this competition contains text that may be considered profane, vulgar, or offensive.

## Evaluation

Submissions are evaluated using F1 between the predicted and expected answers.

F1 is calculated as follows:

$$F1 = 2 * \frac{precision * recall}{precision + recall}$$

where:

$$precision = \frac{TP}{TP + FP}$$

$$recall = \frac{TP}{TP + FN}$$

and:

    - TP is the number of true positives
    - FP is the number of false positives
    - FN is the number of false negatives




## Dataset Description

### What files do I need?

You'll need `train.csv`, `test.csv` and `sample_submission.csv`.

### What should I expect the data format to be?

Each sample in the train and test set has the following information:

    - The `text` of a tweet
    - A `keyword` from that tweet (although this may be blank!)
    - The   location` the tweet was sent from (may also be blank)

### What am I predicting?

You are predicting whether a given tweet is about a real disaster or not. If so, predict a 1. If not, predict a 0.

Files

    - **train.csv** - the training set
    - **test.csv** - the test set
    - **sample_submission.csv** - a sample submission file in the correct format

Columns

    - **id** - a unique identifier for each tweet
    - **text** - the text of the tweet
    - **location** - the location the tweet was sent from (may be blank)
    - **keyword** - a particular keyword from the tweet (may be blank)
    - **target** - in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)


## Link to the competition

https://www.kaggle.com/competitions/nlp-getting-started/overview