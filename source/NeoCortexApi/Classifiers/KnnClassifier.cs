using NeoCortexApi.Encoders;
using NeoCortexApi.Entities;
using System;
using System.Collections.Generic;
using System.Linq;

/*

KNN Classifier Example of How it works:
- Models: Two models used for classification.
- Classified Sequence: [3, 7, 11, 15, 20]
- Unclassified Sequence: [2, 6, 9, 13, 18]

Steps:
1. **LeastValue Method:**
   - For unclassifiedIdx = 2:
     - Comparing 2 against each index in the classifiedSequence:
       |3 - 2| = 1, |7 - 2| = 5, |11 - 2| = 9, |15 - 2| = 13, |20 - 2| = 18.
       LeastValue(classifiedSequence, 2) = 1

   - For unclassifiedIdx = 6:
     - Comparing 6 against each index in the classifiedSequence:
       |3 - 6| = 3, |7 - 6| = 1, |11 - 6| = 5, |15 - 6| = 9, |20 - 6| = 14.
       LeastValue(classifiedSequence, 6) = 1
       
   - Similarly, compute LeastValue for unclassifiedIdx = 9, 13, and 18.

2. **GetDistanceTable Method:**
   - Construct the distanceTable using the results from the LeastValue method:
     - For unclassifiedIdx = 2, 6, 9, 13, 18, the shortest distances are 1 or 2.

3. **Weighted Votes Calculation:**
   - Compute weighted votes based on the inverse of distances:
     - For index 2: 1 / 1 = 1
     - For index 6: 1 / 1 = 1
     - For index 9: 1 / 2 = 0.5
     - For index 13: 1 / 2 = 0.5
     - For index 18: 1 / 2 = 0.5

4. **Overlaps Calculation:**
   - No overlaps found as distances are not zero.

5. **Similarity Scores Calculation:**
   - No overlaps, so no calculations based on overlaps.

6. **Normalization of Weighted Votes:**
   - Normalize the weighted votes and add them to similarity scores:
     - For index 2: 1
     - For index 6: 1
     - For index 9: 0.5
     - For index 13: 0.5
     - For index 18: 0.5

7. **Final Decision:**
   - Order the classifications based on the similarity scores:
     [2, 6, 9, 13, 18] (descending order of calculated scores)

This sequence indicates the predictions in descending order of their similarity scores.
This demonstrates how distances, weighted votes, and similarity scores are calculated and utilized to determine the best classifications for the unclassified points based on their proximity to the classified points.

 */

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

        /// <summary>
        /// Storing the SDR number under the classification. 
        /// </summary>
        //public int ClassificationNo { get; private set; }

        public ClassificationAndDistance(string classification, int distance)
        {
            Classification = classification;
            Distance = distance;
            //ClassificationNo = classificationNo;
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
        /// Learns and updates the K-nearest neighbors (KNN) classifier with new input and associated cells.
        /// </summary>
        /// <param name="input">The input data for learning.</param>
        /// <param name="cells">Array of cells associated with the input.</param>
        public void Learn(TIN input, Cell[] cells)
        {
            // Generate a classification string based on the input
            var classification = GetClassificationFromDictionary(input);
            //Console.WriteLine("Classification: " + classification);
            // Extract indices from the cells array and convert them into an integer array
            int[] cellIndices = cells.Select(idx => idx.Index).ToArray();
            // Print the cell indices
            //Console.WriteLine("Cell Indices: " + string.Join(", ", cellIndices));
            // If the models dictionary does not contain the classification as a key, add it
            if (!models.ContainsKey(classification))
            {
                models[classification] = new List<int[]>();
            }

            // If the sequence of cell indices doesn't exist in the current classification model
            if (!models[classification].Exists(seq => Enumerable.SequenceEqual(seq, cellIndices)))
            {
                // If the count of sequences in the current classification model exceeds the specified value
                if (models[classification].Count > sdrs)
                {
                    // Remove the oldest sequence from the classification model
                    models[classification].RemoveAt(0);
                }

                // Add the new cell indices sequence to the current classification model
                models[classification].Add(cellIndices);
            }
            // Print the final content of the models dictionary
            foreach (var entry in models)
            {
                Console.WriteLine($"Classification: {entry.Key}");
                Console.WriteLine("Cell Indices:");
                foreach (var sequence in entry.Value)
                {
                    Console.WriteLine(string.Join(", ", sequence));
                }
                Console.WriteLine("--------------------------------------");
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
            int neighbors = Math.Min(numberOfNeighbors, models.Values.Sum(x => x.Count)); // Adjust neighbors based on available models

            foreach (var model in models)
            {
                foreach (var sequence in model.Value)
                {
                    var distanceTable = GetDistanceTable(sequence, unclassifiedSequences);

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
                            mappedElements[kvp.Key].Add(new ClassificationAndDistance(classificationKey, distanceAsInt)); // Pass the correct arguments
                        }
                    }
                }
            }
            // For Testing the models by printing
/*            Console.WriteLine("Models:");
            foreach (var model in models)
            {
                Console.WriteLine($"Model: {model.Key}");
                foreach (var sequence in model.Value)
                {
                    Console.WriteLine($"Sequence: {sequence}");
                }
            }

            foreach (var model in models)
            {
                foreach (var sequence in model.Value)
                {
                    var distanceTable = GetDistanceTable(sequence, unclassifiedSequences);
                    Console.WriteLine($"Distance table for {sequence}:");
                    foreach (var kvp in distanceTable)
                    {
                        Console.WriteLine($"Key: {kvp.Key}");
                        foreach (var classificationDistance in kvp.Value)
                        {
                            Console.WriteLine($"Classification: {classificationDistance.Classification}, Distance: {classificationDistance.Distance}");
                        }
                    }
                }
            }*/


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
        private int LeastValue(int[] classifiedSequence, int unclassifiedIdx)
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

            // Calculate similarity scores based on overlaps
            foreach (var overlap in overlaps)
            {
                similarityScores[overlap.Key] = overlap.Value != 0 ? (double)overlap.Value / mapping.Count : 0;
            }

            // Normalize the weighted votes to a similarity score for each class
            foreach (var vote in weightedVotes)
            {
                similarityScores[vote.Key] += vote.Value; // Add weighted votes to similarity score
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
        // Example logic to generate a classification from a Dictionary (you may customize this)
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
    }
}