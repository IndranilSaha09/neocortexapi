using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Entities;
using System;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using NeoCortexEntities.NeuroVisualizer;

namespace UnitTestsProject
{
    [TestClass]
    public class KNeighborsClassifierTests
    {
        [Test]
        public void Learn_AddsSequenceToModel()
        {
            // Arrange
            var classifier = new KNeighborsClassifier<Dictionary<string, object>, string>();
            var input = new Dictionary<string, object>();
            var cells = new[] { new Cell(5), new Cell(10), new Cell(15), new Cell(20), new Cell(25) };

            // Act
            classifier.Learn(input, cells);

            // Assert
            Assert.AreEqual(1, classifier.GetModelsCount());
            // Add more assertions as needed
        }

        [Test]
        public void GetPredictedInputValues_ReturnsPredictedValues()
        {
            // Arrange
            var classifier = new KNeighborsClassifier<Dictionary<string, object>, string>();
            var unclassifiedCells = new[] { new Cell(8), new Cell(12), new Cell(18), new Cell(22), new Cell(27) };
            classifier.Learn(new Dictionary<string, object>(), new[] { new Cell(5), new Cell(10), new Cell(15), new Cell(20), new Cell(25) });

            // Act
            var results = classifier.GetPredictedInputValues(unclassifiedCells);

            // Assert
            Assert.AreEqual(1, results.Count);
            // Add more assertions as needed
        }

        [Test]
        public void SetNumberOfNeighbors_ChangesNumberOfNeighbors()
        {
            // Arrange
            var classifier = new KNeighborsClassifier<Dictionary<string, object>, string>();
            var initialNeighbors = classifier.GetNumberOfNeighbors();

            // Act
            classifier.SetNumberOfNeighbors(5);

            // Assert
            Assert.AreEqual(5, classifier.GetNumberOfNeighbors());
            Assert.AreNotEqual(initialNeighbors, classifier.GetNumberOfNeighbors());
        }

        [Test]
        public void SetSDRS_ChangesSDRSValue()
        {
            // Arrange
            var classifier = new KNeighborsClassifier<Dictionary<string, object>, string>();
            var initialSDRS = classifier.GetSDRS();

            // Act
            classifier.SetSDRS(30);

            // Assert
            Assert.AreEqual(30, classifier.GetSDRS());
            Assert.AreNotEqual(initialSDRS, classifier.GetSDRS());
        }

        [Test]
        public void ClearState_RemovesAllModels()
        {
            // Arrange
            var classifier = new KNeighborsClassifier<Dictionary<string, object>, string>();
            classifier.Learn(new Dictionary<string, object>(), new[] { new Cell(5), new Cell(10), new Cell(15), new Cell(20), new Cell(25) });

            // Act
            classifier.ClearState();

            // Assert
            Assert.AreEqual(0, classifier.GetModelsCount());
            // Add more assertions as needed
        }
    }

    [TestFixture]
    public class KNeighborsClassifierInnerTests
    {
        [Test]
        public void Learn_WithMaximumSDRS_RemovesOldestSequence()
        {
            // Arrange
            var classifier = new KNeighborsClassifier<Dictionary<string, object>, string>();
            classifier.SetSDRS(3);

            // Act
            classifier.Learn(new Dictionary<string, object>(), new[] { new Cell(5), new Cell(10), new Cell(15), new Cell(20), new Cell(25) });
            classifier.Learn(new Dictionary<string, object>(), new[] { new Cell(30), new Cell(35), new Cell(40), new Cell(45), new Cell(50) });
            classifier.Learn(new Dictionary<string, object>(), new[] { new Cell(55), new Cell(60), new Cell(65), new Cell(70), new Cell(75) });
            classifier.Learn(new Dictionary<string, object>(), new[] { new Cell(80), new Cell(85), new Cell(90), new Cell(95), new Cell(100) });

            // Assert
            Assert.AreEqual(3, classifier.GetModelsCount());
            Assert.IsFalse(classifier.HasModelWithSequence(new[] { 5, 10, 15, 20, 25 }));
            // Add more assertions as needed
        }

        // Add other tests in a similar manner
    }
}