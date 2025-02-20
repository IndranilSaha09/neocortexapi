﻿// Copyright (c) Damir Dobric. All rights reserved.
// Licensed under the Apache License, Version 2.0. See LICENSE in the project root for license information.

using NeoCortexApi.Entities;
using System.Collections.Generic;

namespace NeoCortexApi.Classifiers
{
    public interface IClassifier<TIN, TOUT>
    {
        void Learn(TIN input, Cell[] output);
        List<ClassifierResult<TIN>> GetPredictedInputValues(Cell[] predictiveCells, short howMany = 1);
        List<ClassifierResult<TIN>> PredictWithSoftmax(Cell[] unclassifiedCells, short howMany = 1);

        void ClearState();
    }
}