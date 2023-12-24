using NeoCortexApi.Encoders;
using NeoCortexApi.Entities;
using System;
using System.Collections.Generic;
using System.Linq;

/*

- Suppose we have a KNN classifier with two models.
- For this example, let's assume we're dealing with a small set of sequences and cells for simplicity:

Example Scenario:
Classified Sequence: [5, 10, 15, 20, 25]
Unclassified Sequence: [8, 12, 18, 22, 27]
Steps:
Calculating Distances:

Compute distances between each point in the unclassified sequence and the classified sequence.
The distances are:
[3, 2, 3, 2, 2] (computed using Math.Abs(classifiedIdx - unclassifiedIdx))
Constructing Distance Table:

The Distance Table could look like this:
{
  8: [3],
  12: [2],
  18: [3],
  22: [2],
  27: [2]
}
Weighted Votes Calculation:

For each unclassified index, select the top neighbors (let's assume numberOfNeighbors = 3 for this example):

For index 8: [3]
For index 12: [2]
For index 18: [3]
For index 22: [2]
For index 27: [2]
Calculate weighted votes based on the inverse of distances:

For index 8: 1 / 3 = 0.33
For index 12: 1 / 2 = 0.5
For index 18: 1 / 3 = 0.33
For index 22: 1 / 2 = 0.5
For index 27: 1 / 2 = 0.5
Overlaps Calculation:

Identify overlaps when distances are 0:
There are no overlaps in this scenario.
Similarity Scores Calculation:

Compute similarity scores (based on overlaps, but in this example, there are none).
Normalization of Weighted Votes:

Normalize the weighted votes and add them to similarity scores.
For index 8: 0.33
For index 12: 0.5
For index 18: 0.33
For index 22: 0.5
For index 27: 0.5
Final Decision:

Order the classifications based on the similarity scores:
[12, 22, 27, 8, 18] (descending order of calculated scores)
This sequence indicates the predictions in descending order of their similarity scores.
This demonstrates how distances, weighted votes, and similarity scores are calculated and utilized to determine the best classifications for the unclassified points based on their proximity to the classified points.
 */

namespace NeoCortexApi.Classifiers
{

    /// <summary>
    /// Extends the foreach method to give out an item and index of type IEnumerable.
    /// </summary>
    public class DefaultDictionary<TKey, TValue> : Dictionary<TKey, TValue> where TValue : new()
    {
        /// <summary>
        /// Returns the default value of the declared type.
        /// i.e var sample = DefaultDictionary[string, int]()
        /// >>> sample['A']
        /// >>> 0
        /// </summary>
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
    /// A generic container class
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
        /// Implementation of the Method for sorting the given generic object.
        /// </summary>
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
    /// Implementation of the KNN algorithm. 
    /// </summary>
    public class KNeighborsClassifier<TIN, TOUT> : IClassifier<TIN, TOUT>
    {
        private Dictionary<string, List<int[]>> models = new Dictionary<string, List<int[]>>();
        private int numberOfNeighbors = 2;
        private int sdrs = 20;

        public void SetNumberOfNeighbors(int k)
        {
            numberOfNeighbors = k;
        }

        public void SetSDRS(int s)
        {
            sdrs = s;
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

            // Extract indices from the cells array and convert them into an integer array
            int[] cellIndices = cells.Select(idx => idx.Index).ToArray();

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
        private List<ClassifierResult<TIN>> SelectBestClassification(Dictionary<int, List<ClassificationAndDistance>> mapping, int howMany, int numberOfNeighbors)
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