# Supervised Learning Algorithms

**Author:** Narges Saeednejad (ns112)

This directory contains implementations of various supervised learning algorithms, all custom-built using NumPy and basic Python libraries.

## Algorithms Implemented

- **Perceptron**: A simple binary classifier implementing the perceptron learning algorithm
- **Linear Regression**: Implementation of ordinary least squares regression with gradient descent optimization
- **Logistic Regression**: Binary classification model using sigmoid function and gradient descent
- **Neural Networks**: Multi-layer perceptron implementation with backpropagation
- **K-Nearest Neighbors (KNN)**: Classification and regression based on nearest training examples
- **Decision Trees**: Implementation of decision and regression trees with information gain and Gini impurity measures
- **Random Forests**: Ensemble method using multiple decision trees to improve prediction accuracy
- **Ensemble Methods**: Additional ensemble techniques including boosting approaches

## Dataset Usage

These algorithms are applied to the ESG (Environmental, Social, and Governance) and Financial Performance dataset within the main `ESG_Financial_Analysis.ipynb` notebook. The analysis demonstrates how various machine learning techniques can be used to predict financial metrics based on ESG factors.

## Usage Example

```python
from Supervised_Learning.linear_regression import LinearRegression

# Create and train model
model = LinearRegression(learning_rate=0.01, iterations=1000)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

## Reproducing Results

Refer to the `ESG_Financial_Analysis.ipynb` notebook for details on data preprocessing, model training, and results analysis using these implementations. 