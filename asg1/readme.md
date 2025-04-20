# Neural Network from Scratch - California Housing Price Prediction

This project implements a simple feedforward neural network from scratch using NumPy, developed as part of a deep learning lab assignment. The model is trained to predict median house values using the California Housing dataset.

---

## Objective

To build and train a basic neural network without using any deep learning frameworks (such as TensorFlow or PyTorch), focusing on:

- Forward pass
- Backpropagation
- Gradient descent optimization
- Activation and loss functions

---

## Dataset

- **Source**: `sklearn.datasets.fetch_california_housing`
- **Samples**: ~20,000
- **Features**: 8 continuous variables (e.g., median income, average rooms)
- **Target**: Median house value (in $100,000s)

---

## Neural Network Architecture

| Layer        | Size  | Activation |
|--------------|-------|------------|
| Input Layer  | 8     | —          |
| Hidden Layer | 10    | ReLU       |
| Output Layer | 1     | Linear     |

---

## Features Implemented

- Data preprocessing with standardization
- Manual weight and bias initialization
- ReLU and Linear activation functions
- MSE loss and its derivative
- Forward propagation
- Backpropagation using gradients
- Weight updates using vanilla Gradient Descent
- Evaluation metrics and visualizations

---

## Evaluation Metrics

The trained model is evaluated on a test set using the following metrics:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

---

## Visualizations

- Training loss curve over epochs
- True vs. predicted values
- Histogram of prediction errors
- Sorted line plot comparing true and predicted values

---