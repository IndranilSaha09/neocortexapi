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

The Softmax-based classification for the unclassified set, with corresponding probabilities, is as follows:

Predicted Class: Class 3
Probability: Approximately 0.5515

Second Predicted Class: Class 2
Probability: Approximately 0.2900

Third Predicted Class: Class 1
Probability: Approximately 0.1585

This outcome indicates that, according to the Softmax-weighted mechanism, the unclassified set is most likely associated with Class 3, followed by Class 2 and then Class 1. The probabilities represent the model's confidence in each classification based on the calculated Softmax weights.


## Architecture of KNN Classifier:

<img width="222" alt="image" src="https://github.com/IndranilSaha09/neocortexapi/assets/52401793/3cd1f1ad-f137-4ebb-8b71-8cabe62a050e">


The implemented architecture combines Hierarchical Temporal Memory (HTM) and a K-nearest neighbors (KNN) classifier for sequence learning. Leveraging HTM functionalities such as CortexLayer and TemporalMemory from the NeoCortexApi namespace, the process involves configuring HTM parameters and employing a ScalarEncoder to transform scalar values into sparse distributed representations (SDRs). The experiment execution follows a sequence, initializing HTM components (SpatialPooler, TemporalMemory), and the KNN classifier. Training begins with the Spatial Pooler (SP), achieving stability before joint training of SP and Temporal Memory (TM) with provided sequences. The KNN classifier operates alongside HTM, utilizing active and winner cells to predict future elements in sequences, complementing HTM's sequence learning capabilities.


## Softmax Weightage Mechanism

In this project, the softmax function is utilized to determine the weightage of different classes in the final output, contributing to effective classification. The softmax operation involves converting a set of weights into a probability distribution, ensuring that the highest probability class receives the maximum weight.

### Softmax Function

The softmax function is mathematically represented as:

![image](https://github.com/IndranilSaha09/neocortexapi/assets/52401793/ea4692e4-aa67-4f0d-b9f1-03b809b6c6e1)


### Weight Calculation

To obtain weights for different classes, the softmax function is applied to the initial weights, resulting in a probability distribution. The weights are then determined based on these probabilities.

### Softmax Normalization

After calculating weights using the softmax function, the weights are normalized to ensure they sum up to 1. This normalization step enhances interpretability and aids in determining the significance of each class in the final decision.

### Output Decision

The final output is influenced by the probabilities assigned to each class. The class with the highest probability is given the maximum weight, influencing the model's decision-making process.

This softmax weightage mechanism is crucial for effective classification and is employed to enhance the interpretability of the model's predictions.


## Approaches of KNN Classifier:

| **Approach 1: Simple Weightage Algorithm** |  |
| --- | --- |
| **Prediction and Weightage Methods** | [GetPredictedInputValues](https://github.com/IndranilSaha09/neocortexapi/blob/master/source/NeoCortexApi/Classifiers/KnnClassifier.cs#L278), [SelectBestClassification](https://github.com/IndranilSaha09/neocortexapi/blob/master/source/NeoCortexApi/Classifiers/KnnClassifier.cs#L444) |
| **Metrics Available** | [LeastValue](https://github.com/IndranilSaha09/neocortexapi/blob/master/source/NeoCortexApi/Classifiers/KnnClassifier.cs#L329), [ComputeCosineSimilarity](https://github.com/IndranilSaha09/neocortexapi/blob/master/source/NeoCortexApi/Classifiers/KnnClassifier.cs#L374) |
| **Distance Table Methods** | [GetDistanceTable](https://github.com/IndranilSaha09/neocortexapi/blob/master/source/NeoCortexApi/Classifiers/KnnClassifier.cs#L348), [GetDistanceTableforCosine](https://github.com/IndranilSaha09/neocortexapi/blob/master/source/NeoCortexApi/Classifiers/KnnClassifier.cs#L403) |

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


