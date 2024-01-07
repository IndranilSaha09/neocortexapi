# Investigate and Implement KNN Classifier

## Overview

The `KNeighborsClassifier` is an implementation of the K-nearest neighbors (KNN) algorithm that utilizes Softmax normalization and cosine similarity distance metrics. It provides functionality for classification tasks based on the proximity of unclassified sequences to classified sequences in high-dimensional space.


## Approaches of KNN Classifier:

### Approach 1: Simple Weightage Algorithm:
Utilizes the `GetPredictedInputValues` method in conjunction with `SelectBestClassification`, implementing a simple weightage algorithm.

#### Measure metrics available: 
`LeastValue` and `ComputeCosineSimilarity`

#### Distance Table Methods: 
`GetDistanceTable` and `GetDistanceTableforCosine`

### Approach 2: SoftMax Algorithm:
Employs the `PredictWithSoftmax` method alongside `CalculateSoftmaxWeights`, representing a SoftMax algorithm.

#### Measure metric available: 
`ComputeCosineSimilarity`

#### Distance Table Method: 
`GetDistanceTableforCosine`

#### Method for Probability: 
`Softmax`

| **Approach 1: Simple Weightage Algorithm** |  |
| --- | --- |
| **Utilized Methods** | `GetPredictedInputValues`, `SelectBestClassification` |
| **Metrics Available** | `LeastValue`, `ComputeCosineSimilarity` |
| **Distance Table Methods** | `GetDistanceTable`, `GetDistanceTableforCosine` |

| **Approach 2: SoftMax Algorithm** |  |
| --- | --- |
| **Utilized Methods** | `PredictWithSoftmax`, `CalculateSoftmaxWeights` |
| **Metrics Available** | `ComputeCosineSimilarity` |
| **Distance Table Method** | `GetDistanceTableforCosine` |
| **Probability Method** | `Softmax` |

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



