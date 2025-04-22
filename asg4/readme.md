# SMS Spam Classification using Machine Learning

## Overview

This project implements a text classification model to classify SMS messages as either **spam** or **ham** (non-spam). We use Natural Language Processing (NLP) techniques for preprocessing, such as tokenization, stopword removal, stemming, and lemmatization. The model is then trained and evaluated using various metrics like accuracy, precision, recall, and F1-score.

## Dataset

The dataset used in this project is the **SMS Spam Collection Dataset** from Kaggle. The dataset contains 5,574 SMS messages labeled as either "spam" or "ham."

[Dataset Link: SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

## Project Components

1. **Data Preprocessing**:
   - The dataset is cleaned and processed by converting text to lowercase, tokenizing it, removing stopwords, and applying both stemming and lemmatization.

2. **Text Vectorization**:
   - We use the **TF-IDF** (Term Frequency-Inverse Document Frequency) method to convert text data into numerical vectors suitable for training the model.

3. **Model Building**:
   - A **Logistic Regression** classifier is used to build the text classification model.

4. **Model Evaluation**:
   - The model is evaluated using accuracy, precision, recall, F1-score, and a confusion matrix.
