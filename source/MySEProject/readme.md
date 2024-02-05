# Investigate and Implement KNN Classifier

## Implementation

The `KNeighborsClassifier` is an implementation of the K-nearest neighbors (KNN) algorithm that is designed and integrated with the Neocortex API. It takes in a sequence of values and preassigned labels to train the model. Once the model (a Dictionary mapping of labels to their sequences) is trained the user can give unclassified sequence that needs to be labeled.

## Example

In this project, lets assume three distinct sequences labeled as Category X, Category Y, and Category Z are trained using the `Learn` method, which feeds the model with classified sequences represented by TIN input and associated Cell[] cells. Subsequently, the K-Nearest Neighbors (KNN) implementation predicts labels for unclassified sequences in the subsequent stage of the pipeline using the `GetPredictedInputValues` method, returning a list of `ClassifierResult` objects. This approach facilitates sequence classification with the flexibility to customize the number of neighbors considered and the option to apply Softmax normalization for enhanced prediction accuracy.

#### Step 1: Defining Labeled Sequences

In this step, you should define a set of labeled sequences that will be used to train the KNN classifier. Each sequence should be associated with a unique label. For example:

```bash
models = {
    "Category X" : [[5, 8, 12, 15, 19, 21, 24], [7, 9, 11, 14, 16, 20, 23]],
    "Category Y" : [[3, 6, 9, 13, 18, 22, 25], [4, 7, 10, 12, 15, 19, 21]],
    "Category Z" : [[2, 4, 7, 11, 14, 17, 20], [1, 3, 8, 10, 13, 17, 22]]
}
```
#### Step 2: Unclassified Sequence
You should also have an unclassified sequence for which you want to determine the label. For example:

```bash
unclassified = [5, 8, 12, 15, 19, 21, 24]
```

#### Step 3: Training the KNN Model

Using your implementation, feed the labeled sequences (models) into the KNN classifier using a method like Learn. The KNN model will `learn` from these sequences and store them for future classification.

#### Step 4: Classifying the Unclassified Sequence
Next, use the KNN classifier's GetPredictedInputValues method to predict the label for the unclassified sequence. You can specify how many nearest neighbors to consider (e.g., 3) during the classification.

#### Step 5: Verdict
The result of the classification will be a list of ClassifierResult objects. These objects contain the predicted labels and potentially additional information about the prediction.

In this example, the predicted labels for the unclassified sequence may be as follows:

"Category X" being the closest match.
"Category Y" as the next closest match.
And so on...

`PredictedInputValues=[Category X,Category Y,....]`

The output is a list of `ClassifierResult` objects that provide a verdict on the label prediction.

## Approaches of KNN Classifier:

| **Approach 1: Simple Weightage Algorithm Uses** |  |
| --- | --- |
| **Prediction Method** | [GetPredictedInputValues](https://github.com/IndranilSaha09/neocortexapi/blob/master/source/NeoCortexApi/Classifiers/KnnClassifier.cs#L196) |
| **Weightage Method** | [SelectBestClassification](https://github.com/IndranilSaha09/neocortexapi/blob/master/source/NeoCortexApi/Classifiers/KnnClassifier.cs#L379) |
| **Metrics Available** | [LeastValue](https://github.com/IndranilSaha09/neocortexapi/blob/master/source/NeoCortexApi/Classifiers/KnnClassifier.cs#L252) |
| **Distance Table Method** | [GetDistanceTable](https://github.com/IndranilSaha09/neocortexapi/blob/master/source/NeoCortexApi/Classifiers/KnnClassifier.cs#L271) |


| **Approach 2: SoftMax Algorithm Uses** |  |
| --- | --- |
| **Prediction Method** | [PredictWithSoftmax](https://github.com/IndranilSaha09/neocortexapi/blob/master/source/NeoCortexApi/Classifiers/KnnClassifier.cs#L490) (used inside `GetPredictedInputValues`) |
| **Weightage Method** | [CalculateSoftmaxWeights](https://github.com/IndranilSaha09/neocortexapi/blob/master/source/NeoCortexApi/Classifiers/KnnClassifier.cs#L586) |
| **Metrics Available** | [ComputeCosineSimilarity](https://github.com/IndranilSaha09/neocortexapi/blob/master/source/NeoCortexApi/Classifiers/KnnClassifier.cs#L309) |
| **Distance Table Method** | [GetDistanceTableforCosine](https://github.com/IndranilSaha09/neocortexapi/blob/master/source/NeoCortexApi/Classifiers/KnnClassifier.cs#L338)  |
| **Probability Method** | [Softmax](https://github.com/IndranilSaha09/neocortexapi/blob/master/source/NeoCortexApi/Classifiers/KnnClassifier.cs#L558) |


**To enable the use of SoftMax algorithm, follow these steps:**

1. Navigate to [MultiSequenceLearning.cs](https://github.com/IndranilSaha09/neocortexapi/blob/master/source/Samples/NeoCortexApiSample/MultisequenceLearning.cs#L101)
2. Locate the following line of code:

```csharp
IClassifier<string,ComputeCycle>cls = new KNeighborsClassifier<string,ComputeCycle>(useSoftmax:false);
```

3. Change it to:
```csharp
IClassifier<string,ComputeCycle>cls = new KNeighborsClassifier<string, ComputeCycle>(useSoftmax:true);
```


## Getting Started:

Go to the `..\neocortexapi\source\Samples` folder which is one folder above and inside where a folder names NeoCortexApiSample is present.
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


