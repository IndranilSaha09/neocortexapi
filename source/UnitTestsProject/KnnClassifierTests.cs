using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Entities;
using NeoCortexEntities.NeuroVisualizer;
using System.Collections.Generic;

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

    }
}
