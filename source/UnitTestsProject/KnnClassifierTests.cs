using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Entities;
using NeoCortexEntities.NeuroVisualizer;
using System.Collections.Generic;
using System.Linq;

namespace UnitTestsProject
{
    [TestClass]
    public class KnnClassifierTests<TInput, TOutput>
    {
        private Dictionary<string, List<int[]>> models;
        private KNeighborsClassifier<int[], string> knnClassifier;

        [TestInitialize]
        public void Setup()
        {
            models = new Dictionary<string, List<int[]>>();
            knnClassifier = new KNeighborsClassifier<int[], string>();
        }

        [TestMethod]
        public void GetPredictedInputValues_ReturnsEmptyList_WhenUnclassifiedCellsAreEmpty()
        {
            // Arrange
            Cell[] unclassifiedCells = new Cell[] { };

            // Act
            var result = knnClassifier.GetPredictedInputValues(unclassifiedCells);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result.Count);
        }



        [TestMethod]
        public void GetPredictedInputValues_ReturnsEmptyList_WhenNoModelsAvailable()
        {
            // Arrange
            var unclassifiedCells = new Cell[]
            {
                new Cell(0, 1, 0, CellActivity.ActiveCell),
                new Cell(1, 2, 1, CellActivity.ActiveCell),
                // Add more cells as needed for the unclassified sequence
            };

            // Act
            var result = knnClassifier.GetPredictedInputValues(unclassifiedCells, 3);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result.Count);
        }

        /// <summary>
        /// Here we are checking if cell count is zero will we get any kind of exception.
        /// </summary>
        [TestMethod]
        [TestCategory("Prod")]
        public void NoExceptionIfCellsCountIsZero()
        {
            var cells = new Cell[] { };
            var res = knnClassifier.GetPredictedInputValues(cells, 3);
            Assert.AreEqual(res.Count, 0, $"{res.Count} != 0");
        }

        [TestMethod]
        public void GetPredictedInputValues_ReturnsCorrectNumberOfPredictions()
        {
            // Arrange
            var unclassifiedCells = new Cell[]
            {
                new Cell(0, 0, 0, CellActivity.ActiveCell),
                new Cell(1, 1, 1, CellActivity.ActiveCell),
                // Add more cells as needed for the unclassified sequence
            };

            // Learn some data to make predictions
            var input = new int[] { 5, 10, 15, 20, 25 };
            var cells = new Cell[]
            {
                new Cell(0, 0, 0, CellActivity.ActiveCell),
                new Cell(1, 1, 1, CellActivity.ActiveCell),
                // Add more cells as needed for the input sequence
            };
            knnClassifier.Learn(input, cells);

            // Act
            var numberOfPredictions = 3; // Number of predictions expected
            var result = knnClassifier.GetPredictedInputValues(unclassifiedCells, (short)numberOfPredictions);

            // Assert
            // Check if the expected count of predicted values is returned
            Assert.AreEqual(numberOfPredictions, result.Count);
        }

    }
}
