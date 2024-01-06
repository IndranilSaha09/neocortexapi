# Investigate and Implement KNN Classifier

## Overview

The `KNeighborsClassifier` is an implementation of the K-nearest neighbors (KNN) algorithm that utilizes Softmax normalization and cosine similarity distance metrics. It provides functionality for classification tasks based on the proximity of unclassified sequences to classified sequences in high-dimensional space.

## Approaches of KNN Classifier:

### Approach 1: Simple Weightage Algorithm:
Utilizes the `GetPredictedInputValues` method in conjunction with `SelectBestClassification`, implementing a simple weightage algorithm.

#### Measure metrics available: 
`LeastValue` and `ComputeCosineSimilarity`.

#### Distance Table Methods: 
`GetDistanceTable` and `GetDistanceTableforCosine`

### Approach 2: SoftMax Algorithm:
Employs the `PredictWithSoftmax` method alongside `CalculateSoftmaxWeights`, representing a SoftMax algorithm.

#### Measure metric available: 
`ComputeCosineSimilarity`.

#### Distance Table Method: 
`GetDistanceTableforCosine`

#### Method for Probability: 
`Softmax`


## KNN Classifier Methods:

#### `Learn` Method

The `Learn` method updates the models with new input and associated cell indices. It associates input values with cell indices in the models.

#### `GetPredictedInputValues` Method

Predicts input values by computing distances from sequences in multiple models. It selects the best classification based on distances and specified criteria.

#### `LeastValue` Method
Finds the smallest difference between a single value and a sequence of values.

#### `GetDistanceTable` Method
Generates a dictionary mapping the unclassified sequence index to the shortest distance between classified and unclassified sequences.

#### `ComputeCosineSimilarity` Method
Calculates the cosine similarity between two sets (classified and unclassified).

#### `GetDistanceTableforCosine` Method
Computes the cosine similarity between a classified sequence and an unclassified sequence and generates a distance table.

#### `SelectBestClassification` Method
Selects the best classification results based on similarity scores and weighted votes.

#### `GetPredictedInputValues` Method
Predicts input values using K-nearest neighbors (KNN) distances and similarity scores.

#### `PredictWithSoftmax` Method
Predicts classification using K-nearest neighbors distances and applies Softmax normalization for probabilities.

#### `Softmax` Method
Normalizes weights into probabilities across classes using the Softmax function.

#### `CalculateSoftmaxWeights` Method
Computes Softmax-like weights based on distances and classifies them.

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



