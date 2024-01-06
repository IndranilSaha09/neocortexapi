# KNeighborsClassifier with Softmax and Cosine Similarity

## Overview

The `KNeighborsClassifier` is an implementation of the K-nearest neighbors (KNN) algorithm that utilizes Softmax normalization and cosine similarity distance metrics. It provides functionality for classification tasks based on the proximity of unclassified sequences to classified sequences in high-dimensional space.

## Features

### Core Functionality

- **Learn Method:** Adds new input values and associated cell indices to update the models.
- **GetPredictedInputValues Method:** Predicts input values using KNN distances and similarity scores.
- **ClearState Method:** Clears the stored models.

### Softmax and Cosine Similarity

- **PredictWithSoftmax Method:** Predicts classifications using KNN distances and Softmax normalization for probabilities.
- **ComputeCosineSimilarity Method:** Calculates cosine similarity between two sets of sequences.
- **Softmax Method:** Normalizes weights into probabilities using the Softmax function.
- **CalculateSoftmaxWeights Method:** Computes Softmax-like weights based on distances for classification.

## Usage

### Learning and Prediction

