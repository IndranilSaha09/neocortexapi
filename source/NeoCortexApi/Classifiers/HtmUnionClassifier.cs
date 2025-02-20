﻿// Copyright (c) Damir Dobric. All rights reserved.
// Licensed under the Apache License, Version 2.0. See LICENSE in the project root for license information.
using NeoCortexApi.Entities;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NeoCortexApi.Classifiers
{
    public class HtmUnionClassifier<TIN, TOUT> : IClassifier<TIN, TOUT>
    {
        private Dictionary<TIN, int[]> m_ActiveMap = new Dictionary<TIN, int[]>();

        public TIN GetPredictedInputValue(Cell[] predictiveCells)
        {
            int result = 0;
            dynamic charOutput = null;
            int[] arr = new int[predictiveCells.Length];
            for (int i = 0; i < predictiveCells.Length; i++)
            {
                arr[i] = predictiveCells[i].Index;
            }
            foreach (var key in m_ActiveMap.Keys)
            {
                if (result < PredictNextValue(arr, m_ActiveMap[key]))
                {
                    result = PredictNextValue(arr, m_ActiveMap[key]);
                    charOutput = key as string;
                }
            }
            return (TIN)charOutput;
        }

        public List<ClassifierResult<TIN>> GetPredictedInputValues(Cell[] predictiveCells, short howMany = 1)
        {
            throw new System.NotImplementedException();
        }
        public void ClearState()
        {
            // Implementation to clear or reset the classifier's state
            m_ActiveMap.Clear();
        }

        public List<ClassifierResult<TIN>> PredictWithSoftmax(Cell[] predictiveCells, short howMany = 1)
        {
            // Implementation for predicting using the Softmax approach
            // This is just a conceptual placeholder; you'll need to define how softmax prediction should work with your classifier
            throw new NotImplementedException();
        }


        public void Learn(TIN input, Cell[] activeCells, bool learn)
        {
            if (learn == true)
            {
                int[] unionArray;
                int[] cellAsInt = new int[activeCells.Length];
                for (int i = 0; i < activeCells.Length; i++)
                {
                    cellAsInt[i] = activeCells[i].Index;
                }
                if (!m_ActiveMap.TryGetValue(input, out unionArray))
                {
                    m_ActiveMap.Add(input, cellAsInt);
                    return; // or whatever you want to do
                }
                else
                {
                    m_ActiveMap[input] = GetUnionArr(cellAsInt, m_ActiveMap[input]);
                }
            }
        }

        public void Learn(TIN input, Cell[] output)
        {
            throw new System.NotImplementedException();
        }

        private int[] GetUnionArr(int[] prevCells, int[] currCells)
        {
            return prevCells.Union(currCells).ToArray();
        }

        private int PredictNextValue(int[] activeArr, int[] predictedArr)
        {
            var same = predictedArr.Intersect(activeArr);

            return same.Count();
        }
    }
}