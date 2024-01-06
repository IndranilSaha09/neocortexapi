# Investigate and Implement KNN Classifier

## Overview

The `KNeighborsClassifier` is an implementation of the K-nearest neighbors (KNN) algorithm that utilizes Softmax normalization and cosine similarity distance metrics. It provides functionality for classification tasks based on the proximity of unclassified sequences to classified sequences in high-dimensional space.

## Features

## Core Functionality:

- **Learn Method:** Adds new input values and associated cell indices to update the models.
- **GetPredictedInputValues Method:** Predicts input values using KNN distances and similarity scores.
- **ClearState Method:** Clears the stored models.

## Softmax and Cosine Similarity:

### Approach 1: Simple Weightage Algorithm

- Use `GetPredictedInputValues` with `SelectBestClassification`, a simple weightage algorithm.
- Measure metrics: `LeastValue` or `ComputeCosineSimilarity`.

### Approach 2: SoftMax Algorithm

- Use `PredictWithSoftmax` with `CalculateSoftmaxWeights`, a SoftMax algorithm.
- Measure metric: `ComputeCosineSimilarity`.

## Methods in Detail:

### Learn Method

The `Learn` method updates the models with new input and associated cell indices. It associates input values with cell indices in the models.

### GetPredictedInputValues Method

Predicts input values by computing distances from sequences in multiple models. It selects the best classification based on distances and specified criteria.

### ClearState Method

Clears the stored models, allowing for a clean slate to update new models.

### PredictWithSoftmax Method

Predicts classifications using KNN distances and Softmax normalization for probabilities. It applies Softmax to distances, converting them into probabilities for classification.

### ComputeCosineSimilarity Method

Calculates cosine similarity between two sets of sequences, providing a measure of similarity between them.

### Softmax Method

Normalizes weights into probabilities using the Softmax function, ensuring a proper probability distribution across classifications.

### CalculateSoftmaxWeights Method

Computes Softmax-like weights based on distances for classification. It employs Softmax to derive normalized weights associated with different classifications.

This README outlines the core functionalities of the K-nearest neighbors (KNN) algorithm, particularly focusing on the methods for learning, prediction, and clearing state. Additionally, it highlights two different approaches - a Simple Weightage Algorithm and a SoftMax Algorithm - each with their suitable measure metrics.


## Getting Started:

Go to the `Samples` folder which is one folder above and inside where a folder names NeoCortexApiSample is present.
From there run the `Program.cs` file to run the KNN Classifier.

```bash
dotnet run --project "../Samples/NeoCortexApiSample/NeoCortexApiSample.csproj"
```

Path to the
Project: [KnnClassifier.cs](https://github.com/IndranilSaha09/neocortexapi/blob/master/source/NeoCortexApi/Classifiers/KnnClassifier.cs)

## Testing

The unit tests are written under the `UnitTestsProject` also one folder above, run the `KnnClassifierTests.cs` for the
unittests.

Path to the Unit
test: [KnnClassifierTests.cs](https://github.com/IndranilSaha09/neocortexapi/blob/master/source/UnitTestsProject/KnnClassifierTests.cs)



