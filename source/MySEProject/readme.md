# Investigate and Implement KNN Classifier

## Overview

The `KNeighborsClassifier` is an implementation of the K-nearest neighbors (KNN) algorithm that utilizes Softmax normalization and cosine similarity distance metrics. It provides functionality for classification tasks based on the proximity of unclassified sequences to classified sequences in high-dimensional space.

## Example

### Vector Representation:
- Classified Set: [1857, 1862, 2126, ... , 4629, 4954]
- Unclassified Set: [1857, 2141, 2212, ... , 4617, 4954]

### Cosine Similarity Calculation:
- Formula: cosine_similarity = dot_product(A, B) / (||A|| * ||B||)
- Dot Product: Calculate the dot product of the Classified and Unclassified Sets.
- Vector Length: Compute the lengths (magnitudes) of the Classified and Unclassified Sets.

### Understanding Cosine Similarity:
- Ranges from 0 to 1.
- Interpretation:
  - 1 implies perfect similarity (identical vectors).
  - 0 indicates no similarity (orthogonal or completely different vectors).
- Consistently computed cosine similarity of 0.8104408984731079 implies a high degree of similarity between the two sets.

### Deriving Distance from Similarity:
- Formula: Distance = (1 - Cosine Similarity) * 100
- Range: 0 to 100.
  - 0 implies perfect similarity.
  - 100 indicates no similarity.

### Distance Table Representation:
- Shows distances between corresponding elements (keys) of the Classified and Unclassified Sets.
- For instance, the distance for the first entry (Key: 1857) is displayed as 18.
- Since cosine similarity remains consistent, calculated distances for each key pair are also the same (18 in this case).

## Softmax Algorithm and Weightage Mechanism:

### Softmax Algorithm:
- Utilizes the Softmax method.
- Assigns probabilities to elements in the Unclassified Set.
- Computes the softmax weights through PredictWithSoftmax and CalculateSoftmaxWeights.

### Weightage Mechanism:
- Softmax assigns weights to elements based on their probability scores.
- Higher probabilities result in higher weights assigned to the corresponding elements in the Unclassified Set.

### Softmax Probability Distribution:
- Probabilities are calculated using Softmax, ensuring a distribution that amplifies high scores and suppresses low ones.
- The weighted probabilities are used to enhance or influence the classification or prediction process.

### Mathematical Example for Softmax:

Given Probabilities: [0.3, 0.5, 0.7, 0.2, 0.6]

Calculating Softmax Probabilities:
- Softmax(0.3) ≈ 0.1913
- Softmax(0.5) ≈ 0.2338
- Softmax(0.7) ≈ 0.2857
- Softmax(0.2) ≈ 0.1730
- Softmax(0.6) ≈ 0.2583

Resulting Softmax Probabilities: [0.1913, 0.2338, 0.2857, 0.1730, 0.2583]

### Weightage Mechanism:
- Higher probabilities result in higher weights assigned to the corresponding elements in the Unclassified Set. In this case, the element with the highest probability (0.7) will have the highest weight, followed by 0.6, 0.5, 0.3, and 0.2.

These calculations using Softmax illustrate the distribution that amplifies higher scores while suppressing lower ones and the subsequent use of these weighted probabilities to influence the classification or prediction process.


## Approaches of KNN Classifier:

| **Approach 1: Simple Weightage Algorithm** |  |
| --- | --- |
| **Prediction and Weightage Methods** | `GetPredictedInputValues`, `SelectBestClassification` |
| **Metrics Available** | `LeastValue`, `ComputeCosineSimilarity` |
| **Distance Table Methods** | `GetDistanceTable`, `GetDistanceTableforCosine` |

| **Approach 2: SoftMax Algorithm**          |  |
| --- | --- |
| **Prediction and Weightage Methods**       | `PredictWithSoftmax`, `CalculateSoftmaxWeights` |
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



