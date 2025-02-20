﻿using NeoCortexApi.Classifiers;
using NeoCortexApi.Entities;
using NeoCortexApi.Network;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeoCortexApi
{
    /// <summary>
    /// Implements the class that helps inferring (prediction). It is typically used by the Sequence Learning algorithm.
    /// After the learning process, the algorithm returns the instance of this class. This class provides a method <see cref="nameof(Predictor.Predict)"/> with a list of input elements.
    /// For every presented input element the predictor tries to predict the next element. The more element provided in a sequence the predictor returns wit the higher score.
    /// For example, assume there ar two sequences: ABCD and ABGHI. By presenting the element B, the predictor is not sure if the next element is C or D. 
    /// When presenting after B the element C, the predictor knows that the next element must be C.
    /// </summary>
    public class Predictor
    {
        private CortexLayer<object, object> layer1;
        private Connections mem;
        private IClassifier<string, ComputeCycle> classifier { get; set; }

        private Connections connections { get; set; }

        private CortexLayer<object, object> layer { get; set; }

        //private HtmClassifier<string, ComputeCycle> classifier { get; set; }

        /// <summary>
        /// Initializes the predictor functionality.
        /// </summary>
        /// <param name="layer">The HTM Layer.</param>
        /// <param name="connections">The HTM memory in the learned state.</param>
        /// <param name="classifier">The classifier that contains the state of learned sequences.</param>

        public Predictor(CortexLayer<object, object> layer, Connections connections, IClassifier<string, ComputeCycle> classifier)
        {
            this.connections = connections;
            this.layer = layer;
            this.classifier = classifier;
        }
        /// <summary>
        /// Starts predicting of the next subsequences.
        /// </summary>
        public void Reset()
        {
            if (this.layer != null)
            {
                var tmModule = this.layer.HtmModules.FirstOrDefault(m => m.Value is TemporalMemory);
                if (tmModule.Value != null)
                {
                    var tm = (TemporalMemory)tmModule.Value;
                    tm.Reset(this.connections);
                }
                else
                {
                    // Handle the case when TemporalMemory module is not found
                    throw new InvalidOperationException("TemporalMemory module not found.");
                }
            }
            else
            {
                // Handle the case when layer is null
                throw new InvalidOperationException("Layer is null.");
            }
        }



        /// <summary>
        /// Predicts the list of next expected elements.
        /// </summary>
        /// <param name="input">The element that will cause the next expected element.</param>
        /// <returns>The list of expected (predicting) elements.</returns>
        public List<ClassifierResult<string>> Predict(double input)
        {
            var lyrOut = this.layer.Compute(input, false) as ComputeCycle;

            List<ClassifierResult<string>> predictedInputValues = this.classifier.GetPredictedInputValues(lyrOut.PredictiveCells.ToArray(), 3);

            return predictedInputValues;
        }
    }
}
