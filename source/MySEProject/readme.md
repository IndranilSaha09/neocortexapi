# Investigate and Implement KNN Classifier

## Implementation

The `KNeighborsClassifier` is an implementation of the K-nearest neighbors (KNN) algorithm that is designed and integrated with the Neocortex API. It provides functionality for classification tasks based on the proximity of unclassified sequences to classified sequences in high-dimensional space.


## Example

- **Classified Set**: [1857, 1862, 2126, 4629, 4954]
- **Unclassified Set**: [1857, 2141, 2212, 4617, 4954]
- **Cosine Similarity**: 0.8104
- **Distance from Similarity**: 19

#### Cosine Similarity Calculation:
Cosine similarity measures the similarity between two non-zero vectors:

Cosine Similarity = dot product(A, B) / (||A|| * ||B||)

In this case, Cosine Similarity = 0.8104

#### Distance Table for Unclassified Element 1857:
- Class 1: Distance = 5
- Class 2: Distance = 7
- Class 3: Distance = 9

#### Softmax Weight Calculation:
Using `CalculateSoftmaxWeights` method with softness parameter = 0.5:

- Class 1: Weight ≈ -3.184
- Class 2: Weight ≈ -2.884
- Class 3: Weight ≈ -2.584

#### Softmax Normalization:
Applying Softmax method to weights:

- Class 1: Probability ≈ 0.1585
- Class 2: Probability ≈ 0.2900
- Class 3: Probability ≈ 0.5515

#### Weightage Mechanism:
Based on probabilities:

- Class 3 (Probability ≈ 0.5515) - Highest weight.
- Class 2 (Probability ≈ 0.2900) - Second-highest weight.
- Class 1 (Probability ≈ 0.1585) - Lowest weight.



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


## Updated Files for Dependencies

1. [Program.cs](https://github.com/IndranilSaha09/neocortexapi/blob/master/source/Samples/NeoCortexApiSample/Program.cs)
2. [IClassifier.cs](https://github.com/IndranilSaha09/neocortexapi/blob/master/source/NeoCortexApi/Classifiers/IClassifier.cs)
3. [HTMUnionClassifier.cs](https://github.com/IndranilSaha09/neocortexapi/blob/master/source/NeoCortexApi/Classifiers/HtmUnionClassifier.cs)
4. [SDRClassifier](https://github.com/IndranilSaha09/neocortexapi/blob/master/source/NeoCortexApi/Classifiers/SDRClassifier.cs)
5. [Predictor.cs](https://github.com/IndranilSaha09/neocortexapi/blob/master/source/NeoCortexApi/Predictor.cs)

## References

1.https://github.com/UniversityOfAppliedSciencesFrankfurt/LearningApi/blob/f713a28984e8f3115952c54cd9d60d53faa76ffe/LearningApi/src/MLAlgorithms/AnomDetect.KMeans/KMeansAlgorithm.cs

2.https://youtu.be/iUrqokeC7ec?si=q3d0T4R4YopnTfp7&t=614

3.https://hopding.com/sdr-classifier

4.https://discourse.numenta.org/t/how-the-sdr-classifier-works/1481/7

5.https://www.youtube.com/watch?v=iUrqokeC7ec


