# 🧠 Machine Learning Algorithms from Scratch

## 📖 Overview

This repository showcases my understanding of the mathematical foundations of machine learning. Instead of relying on high-level APIs like `scikit-learn` or `TensorFlow`, I have implemented fundamental algorithms from the ground up using **Python** and **NumPy**.

The goal of this project is to demonstrate:
* Deep understanding of **vectorization** and linear algebra.
* Knowledge of **optimization techniques** (Gradient Descent).
* Ability to translate mathematical formulas into efficient code.
* Clean, object-oriented software design.

## 🛠️ Tech Stack
* **Python**: Core logic.
* **NumPy**: Matrix operations and vectorization.
* **Matplotlib/Seaborn**: For visualizing decision boundaries and regression lines (in the notebooks).

## 📂 Implementation Details
### 1. Linear Regression

An implementation of Ordinary Least Squares (OLS) using Gradient Descent to minimize the Mean Squared Error (MSE).Shutterstock Explore 
* **Key Concepts**:
    * **Hypothesis function**: $y = mx + b$ (or $y = w^Tx + b$)
    * **Cost Function**: Mean Squared Error (MSE).
    * **Optimization**: Batch Gradient Descent.
* **Features**: Supports single and multiple linear regression.
### 2. Logistic Regression
A binary classification algorithm that estimates the probability that a given instance belongs to a specific class.

* **Key Concepts**:
    * **Activation Function**: Sigmoid Function $\sigma(z) = \frac{1}{1 + e^{-z}}$.
    * **Cost Function**: Binary Cross-Entropy (Log Loss).
    * **Decision Boundary**: Thresholding at 0.5.
### 3. K-Nearest Neighbors (KNN) - Classifier
A non-parametric, lazy learning algorithm used for classification. It assigns a class based on the majority vote of its neighbors.
* **Key Concepts**:
    * **Distance Metric**: Euclidean Distance $\sqrt{\sum_{i=1}^k (x_i - y_i)^2}$.
    * **Voting Mechanism**: Majority vote (Mode).
* **Features**: Configurable $k$ value.

### 4. K-Nearest Neighbors (KNN) - Regressor
Similar to the classifier, but predicts a continuous value based on the average of the neighbors.
* **Key Concepts**:
    * **Distance Metric**: Euclidean Distance.
    * **Prediction Mechanism**: Mean (average) of the $k$ nearest target values.

### 5. K-Means Clustering
An unsupervised learning algorithm that partitions data into $k$ distinct clusters based on feature similarity.
* **Key Concepts**:
    * **Centroid Initialization**: Random selection.
    * **Assignment Step**: Assign points to the nearest centroid.
    * **Update Step**: Recalculate centroids based on the mean of assigned points.
    * **Convergence**: Stops when centroids do not change or max iterations are reached.

## 💻 Usage Example
The implementations follow a Scikit-Learn style API (`.fit()` and `.predict()`).
```Python
import numpy as np
from MLAlgorithms.LinearRegression import LinearRegression

# 1. Generate dummy data
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# 2. Initialize the model
model = LinearRegression(learning_rate=0.01, n_iters=1000)

# 3. Fit the model to the data (Training)
model.fit(X, y)

# 4. Make predictions
predictions = model.predict(X)

print(predictions)
```

## 🗂️ Directory Structure

```
├── data/                  # Datasets used for testing
├── src/
│   ├── __init__.py
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── knn.py             # Contains both Classifier and Regressor classes
│   └── kmeans.py
├── notebooks/             # Jupyter notebooks visualizing the results
├── README.md
└── requirements.txt
```
## 🚀 Future Improvements
* Implement Support Vector Machines (SVM) using the Kernel trick.
* Implement Decision Trees using Information Gain/Entropy.Add regularization (L1 Lasso / L2 Ridge) to the Regression models.
## 🤝 Contact
Feel free to reach out if you have questions about the implementations or the math behind them!