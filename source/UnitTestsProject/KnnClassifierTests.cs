using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Entities;
using NeoCortexEntities.NeuroVisualizer;
using System;
using System.Collections.Generic;
using System.Linq;

namespace UnitTestsProject
{
    /// <summary>
    /// Class containing tests for the K-nearest neighbors (KNN) classifier.
    /// </summary>
    [TestClass]
    public class KnnClassifierTests
    {
        private KNeighborsClassifier<string, Cell> knnClassifier;
        private Dictionary<string, List<double>> sequences; // Dictionary to hold input sequences

        [TestInitialize]
        public void Setup()
        {
            knnClassifier = new KNeighborsClassifier<string, Cell>();
            sequences = new Dictionary<string, List<double>>();
            sequences.Add("S1", new List<double>(new double[] { 0.0, 1.0, 2.0, 3.0, 4.0, 2.0, 5.0 }));
            LearnknnClassifier(); // Train the classifier on setup
        }

        /// <summary>
        /// Test ensuring the method returns an empty list when no models are available
        /// </summary>
        [TestMethod]
        public void Test_GetPredictedInputValues_ReturnsEmptyList_WhenNoModelsAvailable()
        {
            // Arrange: Ensure that there are no models available by initializing without any training data
            sequences.Clear();

            // Reinitialize the knnClassifier object after clearing sequences
            knnClassifier = new KNeighborsClassifier<string, Cell>();

            // Act: Call the GetPredictedInputValues method with unclassified cells
            var unclassifiedCells = new Cell[]
            {
                new Cell(0, 1, 0, CellActivity.ActiveCell),
                new Cell(1, 2, 1, CellActivity.ActiveCell),
                // Add more cells as needed for the unclassified sequence
            };
            var result = knnClassifier.GetPredictedInputValues(unclassifiedCells, 1);

            // Assert: Check that the returned list is empty
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result.Count, "Expected an empty list when no models are available");
        }

        /// <summary>
        /// Test ensuring no exceptions are thrown when the cells count is zero
        /// </summary>
        [TestMethod]
        public void NoExceptionIfCellsCountIsZero()
        {
            // Arrange: Create an empty array of cells
            var cells = new Cell[] { };

            // Act: Call the GetPredictedInputValues method with zero cells
            var res = knnClassifier.GetPredictedInputValues(cells, 3);

            // Assert: Check that the returned list count is zero
            Assert.AreEqual(res.Count, 0, $"{res.Count} != 0");
        }

        /// <summary>
        /// Test ensuring the correct number of predicted input values are returned
        /// </summary>
        [TestMethod]
        public void CheckHowManyOfGetPredictedInputValues_ReturnsCorrectNumberOfPredictions()
        {
            // Arrange: Set the number of predictions expected
            var howMany = 3;

            // Generate mock cells based on predictive activity
            var predictiveCells = getMockCells(CellActivity.PredictiveCell);

            // Act: Call the GetPredictedInputValues method with predictive cells
            var res = knnClassifier.GetPredictedInputValues(predictiveCells.ToArray(), (short)howMany);

            // Assert: Check that the returned list has the expected count of predicted values
            Assert.IsTrue(res.Count == howMany, $"{res.Count} != {howMany}");
        }


        /// <summary>
        /// Tests whether the GetPredictedInputValues method returns the expected results.
        /// </summary>
        [TestMethod]
        public void Test_GetPredictedInputValues_ReturnsExpectedResults_1()
        {
            // Arrange
            knnClassifier.SetNumberOfNeighbors(3);

            // Define classified and unclassified sequences for testing
            int[] classifiedSequence = new int[] { 5, 10, 15, 20, 25 };
            int[] unclassifiedSequence = new int[] { 8, 12, 18, 22, 27 };

            // Learn the classifier with the classified sequence
            var cells = classifiedSequence.Select(idx => new Cell(idx, idx + 1, idx + 2, CellActivity.ActiveCell)).ToArray();
            knnClassifier.Learn("Classified", cells);

            // Act
            var unclassifiedCells = unclassifiedSequence.Select(idx => new Cell(idx, idx + 1, idx + 2, CellActivity.ActiveCell)).ToArray();
            var result = knnClassifier.GetPredictedInputValues(unclassifiedCells, 5); // Set to retrieve top 5 predictions

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(5, result.Count, "Expected 5 predictions");
        }


        /// <summary>
        /// Tests whether the GetPredictedInputValues method returns the expected results using LearnknnClassifier method.
        /// </summary>
        [TestMethod]
        public void Test_GetPredictedInputValues_ReturnsExpectedResults_2()
        {
            // Arrange
            knnClassifier.SetNumberOfNeighbors(3);

            // Learn the classifier using LearnknnClassifier method
            LearnknnClassifier();

            // Define unclassified sequences for testing
            int[] unclassifiedSequence = new int[] { 8, 12, 18, 22, 27 };

            // Act
            var unclassifiedCells = unclassifiedSequence
                .Select(idx => new Cell(idx, idx + 1, idx + 2, CellActivity.ActiveCell))
                .ToArray();
            var result = knnClassifier.GetPredictedInputValues(unclassifiedCells, 5); // Set to retrieve top 5 predictions

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(5, result.Count, "Expected 5 predictions");
        }


        /// <summary>
        /// Test ensuring the GetPredictedInputValues method returns the expected results with varying numbers of predicted values.
        /// </summary>
        [TestMethod]
        public void Test_GetPredictedInputValues_ReturnsCorrectNumberOfPredictions()
        {
            // Arrange
            knnClassifier.SetNumberOfNeighbors(3);

            // Define classified sequences for testing
            int[] classifiedSequence = new int[] { 5, 10, 15, 20, 25 };

            // Learn the classifier with the classified sequence
            var cells = classifiedSequence.Select(idx => new Cell(idx, idx + 1, idx + 2, CellActivity.ActiveCell)).ToArray();
            knnClassifier.Learn("Classified", cells);

            // Act
            var unclassifiedCells = new Cell[]
            {
                new Cell(8, 9, 10, CellActivity.ActiveCell),
                new Cell(12, 13, 14, CellActivity.ActiveCell),
                new Cell(18, 19, 20, CellActivity.ActiveCell),
            };
            var result = knnClassifier.GetPredictedInputValues(unclassifiedCells, 2); // Set to retrieve top 2 predictions

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(2, result.Count, "Expected 2 predictions");
        }



        /// <summary>
        /// Test ensuring the KNN classifier is initialized correctly with default values.
        /// </summary>
        [TestMethod]
        public void Test_K_value_IsInitializedCorrectly()
        {
            // Arrange
            var defaultKnnClassifier = new KNeighborsClassifier<string, Cell>();

            // Act
            var numOfNeighbors = defaultKnnClassifier.SetNumberOfNeighbors(1);

            // Assert
            Assert.AreEqual(1, numOfNeighbors);
        }

        /// <summary>
        /// Test ensuring the sdr value is initialized correctly with assert values.
        /// </summary>
        [TestMethod]
        public void Test_sdr_value_IsInitializedCorrectly()
        {
            // Arrange
            var defaultKnnClassifier = new KNeighborsClassifier<string, Cell>();

            // Act
            var sdr = defaultKnnClassifier.SetSDRS(20);

            // Assert
            Assert.AreEqual(20, sdr);
        }



        /// <summary>
        /// Test method to verify that SelectBestClassification returns an empty list when the mapping is empty.
        /// </summary>
        [TestMethod]
        public void SelectBestClassification_ReturnsEmptyList_WhenMappingIsEmpty()
        {
            // Arrange
            var classifier = new KNeighborsClassifier<string, Cell>();
            var mapping = new Dictionary<int, List<ClassificationAndDistance>>();
            int howMany = 5;
            int numberOfNeighbors = 3;

            // Act
            var result = classifier.SelectBestClassification(mapping, howMany, numberOfNeighbors);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result.Count);
        }



        /// <summary>
        /// Trains the KNN classifier based on the sequences and previous inputs.
        /// </summary>
        private void LearnknnClassifier()
        {
            int maxCycles = 60;

            foreach (var sequenceKeyPair in sequences)
            {
                int maxPrevInputs = sequenceKeyPair.Value.Count - 1;
                List<string> previousInputs = new List<string>();
                previousInputs.Add("-1.0");

                for (int i = 0; i < maxCycles; i++)
                {
                    foreach (var input in sequenceKeyPair.Value)
                    {
                        previousInputs.Add(input.ToString());
                        if (previousInputs.Count > maxPrevInputs + 1)
                            previousInputs.RemoveAt(0);

                        if (previousInputs.Count < maxPrevInputs)
                            continue;

                        string key = GetKey(previousInputs, input, sequenceKeyPair.Key);
                        List<Cell> actCells = getMockCells(CellActivity.ActiveCell);
                        knnClassifier.Learn(key, actCells.ToArray());
                    }
                }
            }
        }


        /// <summary>
        /// Mock the cells data that we get from the Temporal Memory
        /// </summary>
        private List<Cell> getMockCells(CellActivity activity)
        {
            List<Cell> mockCells = new List<Cell>();

            // Generate mock cells based on the specified activity
            // This is just an example - replace this with your logic to create mock cells
            switch (activity)
            {
                case CellActivity.ActiveCell:
                    // Generate mock active cells
                    for (int i = 0; i < 5; i++)
                    {
                        // Add cells with some properties based on the activity
                        Cell cell = new Cell(i, i + 1, i + 2, CellActivity.ActiveCell);
                        mockCells.Add(cell);
                    }
                    break;
                case CellActivity.PredictiveCell:
                    // Generate mock predictive cells
                    for (int i = 0; i < 3; i++)
                    {
                        // Add cells with some properties based on the activity
                        Cell cell = new Cell(i, i + 1, i + 2, CellActivity.PredictiveCell);
                        mockCells.Add(cell);
                    }
                    break;
                // Handle other CellActivity types if needed
                default:
                    // Default behavior if CellActivity is not recognized
                    break;
            }

            return mockCells;
        }

        /// <summary>
        /// Generates a key based on the inputs and sequenceKey.
        /// </summary>
        private string GetKey(List<string> previousInputs, double input, string sequenceKey)
        {
            // Generate a key based on the inputs and sequenceKey
            // This is just an example - replace this with your actual key generation logic
            string key = $"{string.Join("-", previousInputs)}-{input}-{sequenceKey}";

            return key;
        }

        [TestMethod]
        public void LeastValue_ShouldReturnZero_WhenUnclassifiedIdxIsInClassifiedSequence()
        {
            var defaultclassifier = new KNeighborsClassifier<string, Cell>();

            // Arrange
            int[] classifiedSequence = { 1, 3, 5, 7, 9 };
            int unclassifiedIdx = 3;

            // Act
            int result = defaultclassifier.LeastValue(classifiedSequence, unclassifiedIdx);

            // Assert
            Assert.AreEqual(0, result, "The result should be 0 because unclassifiedIdx is in the classifiedSequence.");
        }

        [TestMethod]
        
        public void LeastValue_ShouldReturnShortestDistance_WhenUnclassifiedIdxIsNotInClassifiedSequence()
        {
            var defaultclassifier = new KNeighborsClassifier<string, Cell>();

            // Arrange
            int[] classifiedSequence = { 1, 3, 5, 7, 9 };
            int unclassifiedIdx = 4; // Not present in classifiedSequence

            // Act
            int result = defaultclassifier.LeastValue(classifiedSequence, unclassifiedIdx);

            // Assert
            Assert.AreEqual(1, result, "The result should be the shortest distance, which is 1 in this case.");
        }



        [TestMethod]
        // Test case to ensure the method returns unclassifiedIdx when the classified sequence is empty.
        public void LeastValue_ShouldReturnUnclassifiedIdx_WhenClassifiedSequenceIsEmpty()
        {
            var defaultclassifier = new KNeighborsClassifier<string, Cell>();

            // Arrange
            int[] classifiedSequence = { };
            int unclassifiedIdx = 5;

            // Act
            int result = defaultclassifier.LeastValue(classifiedSequence, unclassifiedIdx);

            // Assert
            Assert.AreEqual(unclassifiedIdx, result, "The result should be unclassifiedIdx because the classifiedSequence is empty.");
        }

        [TestMethod]
        public void LeastValue_ShouldReturnZero_WhenClassifiedSequenceHasSingleElement()
        {
            var defaultclassifier = new KNeighborsClassifier<string, Cell>();

            // Arrange
            int[] classifiedSequence = { 3 };
            int unclassifiedIdx = 3;

            // Act
            int result = defaultclassifier.LeastValue(classifiedSequence, unclassifiedIdx);

            // Assert
            Assert.AreEqual(0, result, "The result should be 0 because unclassifiedIdx is the only element in classifiedSequence.");
        }

        [TestMethod]
        public void LeastValue_ShouldHandleNegativeValues()
        {
            var defaultclassifier = new KNeighborsClassifier<string, Cell>();

            // Arrange
            int[] classifiedSequence = { -5, -3, 0, 2, 4 };
            int unclassifiedIdx = -2;

            // Act
            int result = defaultclassifier.LeastValue(classifiedSequence, unclassifiedIdx);

            // Assert
            Assert.AreEqual(-2, result, "The result should be the shortest distance, which is -2 in this case.");
        }
    }
}
