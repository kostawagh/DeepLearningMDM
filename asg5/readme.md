# LSTM-based Time Series Forecasting and Text Prediction

This repository contains the code and results for three experiments involving LSTM-based models for time series forecasting and text prediction tasks. The experiments are:

1. **Experiment 5.1**: Forecasting Univariate Time Series using LSTM
2. **Experiment 5.2**: Sequence Text Prediction using LSTM
3. **Experiment 5.3**: Sequence Text Classification using LSTM


## Overview

In this project, LSTM (Long Short-Term Memory) models are applied to solve both time series forecasting and text prediction/classification problems. The following experiments were conducted:

### Experiment 5.1: Forecasting Univariate Time Series
This experiment involves forecasting daily minimum temperatures using LSTM. The dataset used for this task is the "Daily Minimum Temperatures in Melbourne" dataset.

### Experiment 5.2: Sequence Text Prediction
This experiment focuses on predicting the next word in a given sequence of text using an LSTM-based model. The English proverbs dataset was used for this task.

### Experiment 5.3: Sequence Text Classification
This experiment involves classifying movie reviews as "positive" or "negative" using LSTM. The IMDB movie review dataset was used.

## Experiment 5.1: Forecasting Univariate Time Series

### Objective
The goal of this experiment was to predict the next value in a univariate time series (temperature data) using an LSTM model.

### Approach
1. **Data Preprocessing**: The time series data was scaled using MinMaxScaler, and sequences of historical data were created to train the LSTM model.
2. **Model Architecture**: A simple LSTM model was defined to predict the next temperature value based on previous values in the sequence.
3. **Evaluation**: The model performance was evaluated using RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error).

### Results
- **RMSE**: 2.34
- **MAE**: 1.87

## Experiment 5.2: Sequence Text Prediction

### Objective
The objective of this experiment was to predict the next word in a sequence of words using an LSTM model.

### Approach
1. **Data Preprocessing**: The text data was tokenized, and input sequences were created by sliding a window of words.
2. **Model Architecture**: A neural network consisting of an embedding layer, LSTM layer, and output layer was used to predict the next word.
3. **Prediction**: The model was used to generate the next word based on a starting phrase.

### Results
- The model successfully predicted the next word in a sequence after training on the English proverbs dataset.

## Experiment 5.3: Sequence Text Classification

### Objective
In this experiment, the goal was to classify text data into categories (positive or negative) using LSTM.

### Approach
1. **Data Preprocessing**: The IMDB dataset was preprocessed by tokenizing the text and padding the sequences to a fixed length.
2. **Model Architecture**: An LSTM model with an embedding layer, LSTM layer, and output layer was used for binary classification.
3. **Evaluation**: The model's accuracy and classification performance were evaluated using metrics like accuracy, F1-score, and confusion matrix.

### Results
- **Accuracy**: 88.2% on the test data.
- **Confusion Matrix**: Successfully predicted positive and negative reviews with high accuracy.
