using NeoCortexApi.Encoders;
using NeoCortexApi.Entities;
using System;
using System.Collections.Generic;
using System.Linq;

<<<<<<< HEAD

=======
>>>>>>> 20a60febbd24a40bf09aa1ed822212a782d01bc5

namespace NeoCortexApi.Classifiers
{

    /// <summary>
    /// Extends the foreach method to provide a default value of type TValue when a key is absent in the dictionary.
    /// </summary>
    public class DefaultDictionary<TKey, TValue> : Dictionary<TKey, TValue> where TValue : new()
    {
        /// <summary>
        /// Retrieves the default value of type TValue for the specified key.
        /// If the key is not present, adds the key with the default value to the dictionary.
        /// <typeparam name="TKey">A key of Generic type.</typeparam>
        /// <typeparam name="TValue">A newly created value of Generic type.</typeparam>
        public new TValue this[TKey key]
        {
            get
            {
                if (!TryGetValue(key, out TValue val))
                {
                    val = new TValue();
                    Add(key, val);
                }

                return val;
            }
            set => base[key] = value;
        }
    }


    /// <summary>
    /// Represents a pair of Classification and Distance for comparison as container.
    /// </summary>
    public class ClassificationAndDistance : IComparable<ClassificationAndDistance>
    {
        /// <summary>
        /// Comparison classification with respect to model data.
        /// </summary>
        public string Classification { get; private set; }

        /// <summary>
        /// Distance with respect to classification of a model data.
        /// </summary>
        public int Distance { get; private set; }


        public ClassificationAndDistance(string classification, int distance)
        {
            Classification = classification;
            Distance = distance;
        }


        /// <summary>
        /// Compares two ClassificationAndDistance instances based on their distance values.
        /// <param name="other">Past object of the implementation for comparison.</param>
        /// <returns>Comparison between past and present object.</returns>
        public int CompareTo(ClassificationAndDistance other)
        {
            if (other == null)
            {
                return 1; // If 'other' is null, this instance is considered greater.
            }

            if (Distance < other.Distance)
            {
                return -1;
            }
            else if (Distance > other.Distance)
            {
                return 1;
            }
            else
            {
                return 0;
            }
        }
    }

    /// <summary>
    /// Implementation of the K-nearest neighbors (KNN) classifier algorithm. 
    /// </summary>
    public class KNeighborsClassifier<TIN, TOUT> : IClassifier<TIN, TOUT>
    {
        private Dictionary<string, List<int[]>> models = new Dictionary<string, List<int[]>>();
        private int numberOfNeighbors = 3;
        private int sdrs = 10;



        public int SetNumberOfNeighbors(int k)
        {
            numberOfNeighbors = k;
            return numberOfNeighbors;
        }

        public int SetSDRS(int s)
        {
            sdrs = s;
            return sdrs;
        }


        /// <summary>
        /// Learn method to update the models with a new input and associated cell indices.
        /// </summary>
        /// <param name="input">The input value for the model.</param>
        /// <param name="cells">Array of cells associated with the input.</param>
        public void Learn(TIN input, Cell[] cells)
        {
            // Retrieve the classification corresponding to the input
            var classification = GetClassificationFromDictionary(input);

            // Convert the cell indices into an integer array
            var cellIndices = cells.Select(idx => idx.Index).ToArray();

            // Update the models based on the classification and cell indices
            UpdateModels(classification, cellIndices);
        }

        /// <summary>
        /// UpdateModels method adds new cell indices to the corresponding classification in the models dictionary.
        /// </summary>
        /// <param name="classification">The classification label.</param>
        /// <param name="cellIndices">Array of cell indices.</param>
        private void UpdateModels(string classification, int[] cellIndices)
        {
            // If the models dictionary does not contain the classification, create a new list
            if (!models.TryGetValue(classification, out var sequences))
            {
                sequences = new List<int[]>();
                models[classification] = sequences;
            }

            // If the sequence doesn't exist in the classification model, add it
            if (!SequenceExists(sequences, cellIndices))
            {
                // Remove the oldest sequence if the count exceeds the specified limit
                RemoveOldestIfExceedsLimit(sequences);

                // Add the new cell indices sequence to the classification model
                sequences.Add(cellIndices);
            }
        }

        /// <summary>
        /// SequenceExists method checks if a target sequence exists in the provided list of sequences.
        /// </summary>
        /// <param name="sequences">List of integer arrays representing sequences.</param>
        /// <param name="targetSequence">Target sequence to check for existence.</param>
        /// <returns>True if the target sequence exists, otherwise False.</returns>
        private bool SequenceExists(List<int[]> sequences, int[] targetSequence)
        {
            return sequences.Any(seq => Enumerable.SequenceEqual(seq, targetSequence));
        }

        /// <summary>
        /// RemoveOldestIfExceedsLimit method checks the count of sequences and removes the oldest sequence if it exceeds the specified limit.
        /// </summary>
        /// <param name="sequences">List of integer arrays representing sequences.</param>
        private void RemoveOldestIfExceedsLimit(List<int[]> sequences)
        {
            // If the count of sequences exceeds the specified limit, remove the oldest sequence
            if (sequences.Count > sdrs)
            {
                sequences.RemoveAt(0); // Remove the oldest sequence
            }
        }



        /// <summary>
        /// Calculates predicted input values using K-nearest neighbors algorithm.
        /// Iterates through each unclassified cell, computes distances from sequences in multiple models,
        /// and selects the best classification based on distances and specified criteria.
        /// </summary>
        /// <param name="unclassifiedCells">Array of unclassified cells.</param>
        /// <param name="howMany">Number of predicted values to return (default = 1).</param>
        /// <returns>List of predicted input values with their similarity scores.</returns>
        public List<ClassifierResult<TIN>> GetPredictedInputValues(Cell[] unclassifiedCells, short howMany = 1)
        {
            if (unclassifiedCells.Length == 0)
                return new List<ClassifierResult<TIN>>();

            var unclassifiedSequences = unclassifiedCells.Select(cell => cell.Index).ToArray();
            var mappedElements = new DefaultDictionary<int, List<ClassificationAndDistance>>();
            int neighbors = Math.Min(numberOfNeighbors, models.Values.Sum(x => x.Count));

            foreach (var model in models)
            {
                foreach (var sequence in model.Value)
                {
                    var distanceTable = GetDistanceTableforCosine(sequence, unclassifiedSequences);

                    foreach (var kvp in distanceTable)
                    {
                        if (!mappedElements.ContainsKey(kvp.Key))
                        {
                            mappedElements[kvp.Key] = new List<ClassificationAndDistance>();
                        }

                        foreach (var classificationDistance in kvp.Value)
                        {
                            int distanceAsInt = classificationDistance.Distance; // Use the distance value
                            string classificationKey = model.Key; // Use the model key as classificationKey
                            mappedElements[kvp.Key].Add(new ClassificationAndDistance(classificationKey, distanceAsInt));
                        }
                    }
                }
            }
          
            foreach (var mappings in mappedElements)
                mappings.Value.Sort(); // Sort values according to distance

            return SelectBestClassification(mappedElements, howMany, neighbors);
        }


        /// <summary>
        /// This method compares a single value with a sequence of values from given sequence.
        /// </summary>
        /// <param name="classifiedSequence">
        /// The active indices from the classified Sequence.
        /// </param>
        /// <param name="unclassifiedIdx">
        /// The active index from the unclassified Sequence.
        /// </param>
        /// <returns>
        /// Returns the smallest value of type int from the list.
        /// </returns>
        public int LeastValue(int[] classifiedSequence, int unclassifiedIdx)
        {
            int shortestDistance = unclassifiedIdx;
            foreach (var classifiedIdx in classifiedSequence)
            {
                var distance = Math.Abs(classifiedIdx - unclassifiedIdx);
                if (shortestDistance > distance)
                    shortestDistance = distance;
            }

            return shortestDistance;
        }

        /// <summary>
        /// This function computes the distances of the unclassified points to the distance of the classified points.
        /// </summary>
        /// <param name="classifiedSequence">Sequence of classified points.</param>
        /// <param name="unclassifiedSequence">Sequence of unclassified points.</param>
        /// <returns>Returns a dictionary mapping of the Unclassified sequence index to the shortest distance.</returns>
        private Dictionary<int, List<ClassificationAndDistance>> GetDistanceTable(int[] classifiedSequence, int[] unclassifiedSequence)
        {
            var distanceTable = new Dictionary<int, List<ClassificationAndDistance>>();

            foreach (var unclassifiedIdx in unclassifiedSequence)
            {
                var shortestDistance = LeastValue(classifiedSequence, unclassifiedIdx);
                if (!distanceTable.ContainsKey(unclassifiedIdx))
                {
                    distanceTable[unclassifiedIdx] = new List<ClassificationAndDistance>();
                }
                distanceTable[unclassifiedIdx].Add(new ClassificationAndDistance("Classification", shortestDistance));
            }

            return distanceTable;
        }


        //--------------------------------------------------------------------------------------------------------------------------------------------------//

        /// <summary>
        /// Computes the cosine similarity between two sets.
        /// </summary>
        /// <param name="classifiedSet">A HashSet representing the classified set of indices.</param>
        /// <param name="unclassifiedSet">A HashSet representing the unclassified set of indices.</param>
        /// <returns>The cosine similarity between the classified and unclassified sets.</returns>
        private double ComputeCosineSimilarity(HashSet<int> classifiedSet, HashSet<int> unclassifiedSet)
        {
            // Calculate the dot product between the classified and unclassified sets
            var dotProduct = classifiedSet.Intersect(unclassifiedSet).Count();

            // Calculate the lengths of the classified and unclassified sets
            var classifiedLength = classifiedSet.Count;
            var unclassifiedLength = unclassifiedSet.Count;

            // Check for edge cases where either of the sets has zero length to avoid division by zero
            if (classifiedLength == 0 || unclassifiedLength == 0)
            {
                return 0.0; // Return 0 if any of the lengths is zero to avoid division by zero
            }

            // Compute the cosine similarity using the dot product and lengths of the sets
            var cosineSimilarity = dotProduct / (Math.Sqrt(classifiedLength) * Math.Sqrt(unclassifiedLength));
            return cosineSimilarity;
        }




        /// <summary>
        /// Computes the cosine similarity between a classified sequence and an unclassified sequence and generates a distance table.
        /// </summary>
        /// <param name="classifiedSequence">Array representing the classified sequence.</param>
        /// <param name="unclassifiedSequence">Array representing the unclassified sequence.</param>
        /// <returns>A dictionary containing the distance table for the unclassified sequence.</returns>
        private Dictionary<int, List<ClassificationAndDistance>> GetDistanceTableforCosine(int[] classifiedSequence, int[] unclassifiedSequence)
        {
            // Create an empty distance table to store classification distances for each unclassified index
            var distanceTable = new Dictionary<int, List<ClassificationAndDistance>>();

            // Convert input arrays to HashSet for faster intersection operations
            var classifiedSet = new HashSet<int>(classifiedSequence);

            // Compute cosine similarity and generate distance table for each unclassified index
            foreach (var unclassifiedIdx in unclassifiedSequence.Distinct())
            {
                var unclassifiedSet = new HashSet<int>(unclassifiedSequence);
                // Create an entry in the distance table if it doesn't exist for the unclassified index
                if (!distanceTable.ContainsKey(unclassifiedIdx))
                {
                    distanceTable[unclassifiedIdx] = new List<ClassificationAndDistance>();
                }

                // Calculate cosine similarity between classified and unclassified sets
                double cosineSimilarity = ComputeCosineSimilarity(classifiedSet, unclassifiedSet);

                // Convert cosine similarity to a distance metric and add to the distance table
                int distance = (int)((1 - cosineSimilarity) * 100); // Assuming cosine similarity is in range [0, 1]
                distanceTable[unclassifiedIdx].Add(new ClassificationAndDistance("Classification", distance));
            }

            return distanceTable;
        }



        //--------------------------------------------------------------------------------------------------------------------------------------------------//


        /// <summary>
        /// Selects the best classification results based on similarity scores and weighted votes.
        /// </summary>
        /// <param name="mapping">A dictionary mapping of indices to lists of ClassificationAndDistance objects.</param>
        /// <param name="howMany">Number of best results to select.</param>
        /// <param name="numberOfNeighbors">Number of neighbors to consider for similarity calculation.</param>
        /// <returns>A list of ClassifierResult objects representing the predicted input values.</returns>
        public List<ClassifierResult<TIN>> SelectBestClassification(Dictionary<int, List<ClassificationAndDistance>> mapping, int howMany, int numberOfNeighbors)
        {
            var weightedVotes = new Dictionary<string, double>(); // Use double for weighted votes
            var overlaps = new Dictionary<string, int>();
            var similarityScores = new Dictionary<string, double>();

            foreach (var key in models.Keys)
            {
                overlaps[key] = 0;
                weightedVotes[key] = 0.0; // Initialize as double for weighted votes
            }

            foreach (var coordinates in mapping)
            {
                var selectedNeighbors = coordinates.Value.Take(numberOfNeighbors); // Select top 'numberOfNeighbors' neighbors

                // Calculate votes based on distances using weights based on distance
                foreach (var value in selectedNeighbors)
                {
                    if (value.Distance == 0)
                    {
                        overlaps[value.Classification]++;
                    }
                    else
                    {
                        // Weighted votes based on distance
                        weightedVotes[value.Classification] += 1.0 / value.Distance;
                    }
                }
            }

            // Normalize the weighted votes before combining with overlap scores
            if (weightedVotes.Any())
            {
                var maxWeightedVote = weightedVotes.Max(v => v.Value);
                foreach (var vote in weightedVotes)
                {
                    weightedVotes[vote.Key] = vote.Value / maxWeightedVote; // Normalize votes to balance contribution
                }
            }
            // Calculate similarity scores based on overlaps
            foreach (var overlap in overlaps)
            {
                similarityScores[overlap.Key] = overlap.Value != 0 ? (double)overlap.Value / mapping.Count : 0;
            }

            // Combine normalized weighted votes with overlap scores
            foreach (var vote in weightedVotes)
            {
                similarityScores[vote.Key] += vote.Value; // Add normalized weighted votes to similarity score
            }

            // Order by similarity scores to make the final decision
            var orderedResults = similarityScores.OrderByDescending(x => x.Value).Select(x => x.Key);

            var result = orderedResults.Select(key => new ClassifierResult<TIN>
            {
                PredictedInput = (TIN)Convert.ChangeType(key, typeof(TIN)),
                Similarity = similarityScores[key],
                NumOfSameBits = overlaps[key] // Consider updating this based on the new approach
            }).ToList();

            return result.Take(howMany).ToList();
        }


        /// <summary>
        /// Generates a classification string from the input object.
        /// </summary>
        /// <param name="input">The input object from which classification is generated.</param>
        /// <returns>A string representing the classification derived from the input.</returns>
        private string GetClassificationFromDictionary(TIN input)
        {
            // Assuming input is a Dictionary<string, object> for demonstration purposes
            var dictionary = input as Dictionary<string, object>;

            if (dictionary != null)
            {
                // Some logic to extract a string representation from the dictionary
                // For example, concatenating dictionary keys or values
                return string.Join("_", dictionary.Keys);
            }

            // Return a default string representation if unable to process
            return input.ToString();
        }

        /// <summary>
        /// Clears the model from all the stored sequences.
        /// </summary>
        public void ClearState()
        {
            models.Clear();
        }



        //------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        //------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        //------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        //------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        //------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        //--------------------------------------CLASSIFY KNN USING SOFTMAX ALGORITHM USING COSINE SIMILARITY DISTANCE METRICS-----------------------------------------------------------


        /// <summary>
        /// Predicts the classification using K-nearest neighbors (KNN) distances and applies Softmax normalization to obtain probabilities.
        /// </summary>
        /// <param name="unclassifiedCells">Array of unclassified cells to predict their classifications.</param>
        /// <param name="howMany">Number of predicted classifications to return (default is 1).</param>
        /// <returns>List of ClassifierResult objects representing the predicted classifications with Softmax probabilities.</returns>
        public List<ClassifierResult<TIN>> PredictWithSoftmax(Cell[] unclassifiedCells, short howMany = 1)
        {
            // If there are no unclassified cells, return an empty list
            if (unclassifiedCells.Length == 0)
                return new List<ClassifierResult<TIN>>();

            // Extract indices of unclassified cells
            var unclassifiedSequences = unclassifiedCells.Select(cell => cell.Index).ToArray();
            // Create a dictionary to store distances and classifications
            var mappedElements = new DefaultDictionary<int, List<ClassificationAndDistance>>();
            // Determine the number of neighbors based on available models
            int neighbors = Math.Min(numberOfNeighbors, models.Values.Sum(x => x.Count));

            // Populate mapped elements with distances and classifications
            foreach (var model in models)
            {
                foreach (var sequence in model.Value)
                {
                    var distanceTable = GetDistanceTableforCosine(sequence, unclassifiedSequences);

                    foreach (var kvp in distanceTable)
                    {
                        if (!mappedElements.ContainsKey(kvp.Key))
                        {
                            mappedElements[kvp.Key] = new List<ClassificationAndDistance>();
                        }

                        foreach (var classificationDistance in kvp.Value)
                        {
                            int distanceAsInt = classificationDistance.Distance;
                            string classificationKey = model.Key;
                            mappedElements[kvp.Key].Add(new ClassificationAndDistance(classificationKey, distanceAsInt));
                        }
                    }
                }
            }

            // Sort values according to distance
            foreach (var mappings in mappedElements)
            {
                mappings.Value.Sort();
            }

            // Calculate softmax weights for each class based on distances
            var softmaxWeights = CalculateSoftmaxWeights(mappedElements, 0.5);

            // Get softmax probabilities from the softmax weights
            var softmaxProbabilities = Softmax(softmaxWeights);

            // Prepare results with softmax probabilities
            var results = softmaxProbabilities.Select(kv => new ClassifierResult<TIN>
            {
                PredictedInput = kv.Key,
                Similarity = kv.Value, // Using softmax probability as similarity score
                                       
            }).ToList();

            return results.Take(howMany).ToList();
        }

        /// <summary>
        /// Normalizes the weights into probabilities across classes using the Softmax function.
        /// </summary>
        /// <param name="weights">Dictionary containing weights associated with different classes.</param>
        /// <returns>Dictionary with Softmax-normalized probabilities for each class.</returns>
        private Dictionary<TIN, double> Softmax(Dictionary<TIN, double> weights)
        {
            // Find the maximum weight for numerical stability
            var maxWeight = weights.Values.Max();

            // Calculate the exponential sum of weights
            var expSum = weights.Values.Sum(w => Math.Exp(w - maxWeight));

            // Normalize weights into probabilities using Softmax formula
            var softmaxProbabilities = new Dictionary<TIN, double>();

            foreach (var kvp in weights)
            {
                double expWeight = Math.Exp(kvp.Value - maxWeight);
                double probability = expWeight / expSum;
                softmaxProbabilities[kvp.Key] = probability;
            }

            return softmaxProbabilities;
        }


        /// <summary>
        /// Computes Softmax-like weights based on distances and classifies them.
        /// </summary>
        /// <param name="distances">Dictionary containing distances between classes.</param>
        /// <param name="softness">The softness parameter used in weight calculation.</param>
        /// <returns>Dictionary with Softmax-like weights classified across classes.</returns>
        private Dictionary<TIN, double> CalculateSoftmaxWeights(DefaultDictionary<int, List<ClassificationAndDistance>> distances, double softness)
        {
            var softmaxWeights = new Dictionary<TIN, double>();

            foreach (var kvp in distances)
            {
                // Calculate the exponential values based on distances for each classification
                var expValues = kvp.Value
                    .GroupBy(classificationDistance => classificationDistance.Classification)
                    .ToDictionary(
                        group => group.Key,
                        group => group.Sum(item => Math.Exp(-Convert.ToDouble(item.Distance) / softness))
                    );

                // Calculate the sum of exponential values
                var expSum = expValues.Values.Sum();
                // Find the maximum exponential value
                var maxWeight = expValues.Values.Max();

                foreach (var softmaxItem in expValues)
                {
                    TIN convertedKey = (TIN)Convert.ChangeType(softmaxItem.Key, typeof(TIN));

                    // Calculate Softmax-like weights
                    if (!softmaxWeights.ContainsKey(convertedKey))
                    {
                        softmaxWeights.Add(convertedKey, Math.Log(expSum) - Math.Log(softmaxItem.Value + Math.Exp(maxWeight - softness)));
                    }
                    else
                    {
                        softmaxWeights[convertedKey] += Math.Log(expSum) - Math.Log(softmaxItem.Value + Math.Exp(maxWeight - softness));
                    }
                }
            }

            return softmaxWeights;
        }


    }
}