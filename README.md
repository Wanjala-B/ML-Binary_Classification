# MNIST Handwritten Digit Classification Project

This repository documents a machine learning project focused on classifying handwritten digits using the MNIST dataset. The goal is to build and evaluate models capable of accurately identifying digits from images, starting with binary classification (detecting a specific digit) and then exploring more complex classifiers.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Binary Classification: Detecting the Digit '5'](#binary-classification-detecting-the-digit-5)
  - [Model Training: `SGDClassifier`](#model-training-sgdclassifier)
  - [Performance Evaluation](#performance-evaluation)
    - [Accuracy with Cross-Validation](#accuracy-with-cross-validation)
    - [Confusion Matrix](#confusion-matrix)
    - [Precision, Recall, and F1-Score](#precision-recall-and-f1-score)
    - [Precision/Recall Trade-off](#precisionrecall-trade-off)
    - [ROC Curve and AUC](#roc-curve-and-auc)
  - [Comparing with `RandomForestClassifier`](#comparing-with-randomforestclassifier)
- [Resources](#resources)

## Project Overview
This project details the process of training and evaluating machine learning models on the MNIST dataset. It covers data loading, initial exploration, data splitting, training binary classifiers, and a thorough evaluation using various metrics and visualizations such as confusion matrices, precision-recall curves, and ROC curves.

## Dataset
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9). Each image is 28x28 pixels, resulting in 784 features. The dataset is loaded using `sklearn.datasets.fetch_openml`.

- **Shape of Data (X):** `(70000, 784)`
- **Shape of Labels (y):** `(70000,)`
- **Data Split:**
  - Training set: First 60,000 instances (`X_train`, `y_train`)
  - Test set: Last 10,000 instances (`X_test`, `y_test`)

An example digit (the first one, a '5') was visualized to ensure correct data loading and representation.

## Binary Classification: Detecting the Digit '5'

### Model Training: `SGDClassifier`
A Stochastic Gradient Descent (SGD) classifier was trained to identify whether a digit is a '5' or not. This involved transforming the original multi-class labels into a binary target (`y_train_5`, `y_test_5`), where `True` indicates a '5' and `False` indicates any other digit.

### Performance Evaluation

#### Accuracy with Cross-Validation
Initial evaluation using 3-fold cross-validation showed an accuracy of approximately 95-96% for the `SGDClassifier` in detecting '5's. However, accuracy alone can be misleading, especially with imbalanced datasets.

*   **`SGDClassifier` Accuracy:** `[0.95035, 0.96035, 0.9604 ]`
*   **`Never5Classifier` Accuracy:** A baseline classifier that always predicts 'not a 5' achieved ~91% accuracy, highlighting the imbalance and the need for more robust metrics.

#### Confusion Matrix
The confusion matrix for the `SGDClassifier` (detecting '5's) on the training set was:

```
[[53892,   687],
 [ 1891,  3530]]
```

- **True Negatives (TN):** 53,892 (Correctly classified non-'5's)
- **False Positives (FP):** 687 (Non-'5's incorrectly classified as '5's)
- **False Negatives (FN):** 1,891 (Actual '5's incorrectly classified as non-'5's)
- **True Positives (TP):** 3,530 (Correctly classified '5's)

#### Precision, Recall, and F1-Score
These metrics provide a more detailed view of the model's performance:

- **Precision:** `0.837` (When the model predicts a '5', it's correct 83.7% of the time).
- **Recall:** `0.651` (The model correctly identifies 65.1% of all actual '5's).
- **F1-Score:** `0.733` (The harmonic mean of precision and recall, balancing both metrics).

#### Precision/Recall Trade-off
A plot of precision and recall against the decision threshold demonstrated the inherent trade-off: as precision increases, recall tends to decrease, and vice-versa. The choice of an optimal threshold depends on the specific requirements of the application (e.g., prioritizing minimizing false positives vs. minimizing false negatives).

#### ROC Curve and AUC
The Receiver Operating Characteristic (ROC) curve plots the True Positive Rate (Recall) against the False Positive Rate. The Area Under the Curve (AUC) summarizes the model's overall discriminative ability.

- **`SGDClassifier` ROC AUC Score:** `0.9605`

An AUC of 0.9605 indicates that the `SGDClassifier` is very good at distinguishing between '5's and non-'5's.

### Comparing with `RandomForestClassifier`
A `RandomForestClassifier` was also trained and evaluated to compare its performance against the `SGDClassifier`.

- **`RandomForestClassifier` ROC AUC Score:** `0.9983`

The `RandomForestClassifier` achieved a significantly higher ROC AUC score, indicating superior performance in this binary classification task. Its ROC curve was observed to be much closer to the top-left corner, signifying a better balance between True Positive Rate and False Positive Rate compared to the `SGDClassifier`.

## Resources
This project leverages several key resources and libraries for machine learning and data analysis:

*   **Scikit-learn:** For data loading, model selection (`SGDClassifier`, `RandomForestClassifier`), cross-validation, and performance metrics.
*   **NumPy:** For numerical operations.
*   **Pandas:** For data manipulation and analysis.
*   **Matplotlib:** For data visualization, including digit display, precision-recall curves, and ROC curves.
*   **MNIST Dataset:** The primary dataset used for handwritten digit classification.
