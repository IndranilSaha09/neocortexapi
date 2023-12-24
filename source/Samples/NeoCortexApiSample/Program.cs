using GemBox.Spreadsheet.Drawing;
using NeoCortexApi;
using NeoCortexApi.Encoders;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using static NeoCortexApiSample.MultiSequenceLearning;

namespace NeoCortexApiSample
{
    class Program
    {
        /// <summary>
        /// This sample shows a typical experiment code for SP and TM.
        /// You must start this code in debugger to follow the trace.
        /// and TM.
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            //
            // Starts experiment that demonstrates how to learn spatial patterns.
            //SpatialPatternLearning experiment = new SpatialPatternLearning();
            //experiment.Run();

            //
            // Starts experiment that demonstrates how to learn spatial patterns.
            //SequenceLearning experiment = new SequenceLearning();
            //experiment.Run();


            //RunMultiSimpleSequenceLearningExperiment();
            RunMultiSequenceLearningExperiment();
            //RunMultiSequenceLearningExperimentWithImage();
        }

        private static void RunMultiSimpleSequenceLearningExperiment()
        {
            Dictionary<string, List<double>> sequences = new Dictionary<string, List<double>>();

            sequences.Add("S1", new List<double>(new double[] { 'a','b','c','d' }));
            sequences.Add("S2", new List<double>(new double[] { 'e','f','g','h'}));

            //
            // Prototype for building the prediction engine.
            MultiSequenceLearning experiment = new MultiSequenceLearning();
            var predictor = experiment.Run(sequences);         
        }


        /// <summary>
        /// This example demonstrates how to learn two sequences and how to use the prediction mechanism.
        /// First, two sequences are learned.
        /// Second, three short sequences with three elements each are created und used for prediction. The predictor used by experiment privides to the HTM every element of every predicting sequence.
        /// The predictor tries to predict the next element.
        /// </summary>
        private static void RunMultiSequenceLearningExperiment()
        {
            Dictionary<string, List<double>> sequences = new Dictionary<string, List<double>>();

            //sequences.Add("S1", new List<double>(new double[] { 0.0, 1.0, 0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0, 3.0, 7.0, 1.0, 9.0, 12.0, 11.0, 12.0, 13.0, 14.0, 11.0, 12.0, 14.0, 5.0, 7.0, 6.0, 9.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0 }));
            //sequences.Add("S2", new List<double>(new double[] { 0.8, 2.0, 0.0, 3.0, 3.0, 4.0, 5.0, 6.0, 5.0, 7.0, 2.0, 7.0, 1.0, 9.0, 11.0, 11.0, 10.0, 13.0, 14.0, 11.0, 7.0, 6.0, 5.0, 7.0, 6.0, 5.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0 }));

            sequences.Add("S1", new List<double>(new double[] { 10.0, 1.0, 2.0, 3.0, 4.0, 2.0, 5.0, }));
            sequences.Add("S2", new List<double>(new double[] { 8.0, 1.0, 2.0, 9.0, 10.0, 7.0, 11.00 }));
            //sequences.Add("S1", new List<double>(new double[] { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h' }));
            //sequences.Add("S2", new List<double>(new double[] { 'i','j','k','l','m','n','o','p' }));
            //
            // Prototype for building the prediction engine.
            MultiSequenceLearning experiment = new MultiSequenceLearning();
            var predictor = experiment.Run(sequences);

            //
            // These list are used to see how the prediction works.
            // Predictor is traversing the list element by element. 
            // By providing more elements to the prediction, the predictor delivers more precise result.
            var list1 = new double[] { 1.0, 2.0, 3.0, 4.0, 2.0, 5.0 };
            var list2 = new double[] { 2.0, 3.0, 4.0 };
            var list3 = new double[] { 8.0, 1.0, 2.0 };

            predictor.Reset();
            PredictNextElement(predictor, list1);

            predictor.Reset();
            PredictNextElement(predictor, list2);

            predictor.Reset();
            PredictNextElement(predictor, list3);
        }
        private static void RunMultiSequenceLearningExperimentWithImage()
        {
            string currentDirectory = AppDomain.CurrentDomain.BaseDirectory;
            string targetDirectoryName = "NeoCortexApiSample"; // Target directory name

            string targetDirectory = currentDirectory;

            while (targetDirectory != null && !targetDirectory.EndsWith(targetDirectoryName))
            {
                targetDirectory = Directory.GetParent(targetDirectory)?.FullName;
            }

            if (targetDirectory != null)
            {
                Console.WriteLine("Target Directory Found: " + targetDirectory);
            }
            else
            {
                Console.WriteLine("Target Directory not found.");
            }

            string folderName = "input_image"; // Folder name containing the images
            string folderPath = Path.Combine(targetDirectory, folderName);


            Console.WriteLine("Folder Path: " + folderPath);

            if (!Directory.Exists(folderPath))
            {
                Console.WriteLine("Folder not found: " + folderPath);
                return;
            }

            List<double[]> pixelSequences = new List<double[]>();

            string[] imageFiles = Directory.GetFiles(folderPath, "*.jpg");

            foreach (string imagePath in imageFiles)
            {
                List<double> pixelSequence = ConvertImageToSequence(imagePath);
                pixelSequences.Add(pixelSequence.ToArray());
            }

            // Print the pixel values for each image
            foreach (var sequence in pixelSequences)
            {
                Console.WriteLine("Sequence:");
                foreach (var pixelValue in sequence)
                {
                    Console.Write($"{pixelValue}, ");
                }
                Console.WriteLine();
            }

            MultiSequenceLearning experiment = new MultiSequenceLearning();
            // Convert the list of arrays to a dictionary or other structure expected by the Run method
            Dictionary<string, List<double>> sequences = new Dictionary<string, List<double>>();
            for (int i = 0; i < pixelSequences.Count; i++)
            {
                sequences.Add($"Sequence_{i + 1}", pixelSequences[i].ToList());
            }

            var predictor = experiment.Run(sequences);
            predictor.Reset();
        }





        private static List<double> ConvertImageToSequence(string imagePath)
        {
            List<double> pixelSequence = new List<double>();

            // Load the image
            Bitmap image = new Bitmap(imagePath);

            // Lock the image data to read its pixel values
            BitmapData bmpData = image.LockBits(new Rectangle(0, 0, image.Width, image.Height), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);

            unsafe
            {
                byte* ptr = (byte*)bmpData.Scan0;

                // Loop through each pixel and convert its RGB values to grayscale
                for (int y = 0; y < bmpData.Height; y++)
                {
                    for (int x = 0; x < bmpData.Width; x++)
                    {
                        int index = y * bmpData.Stride + x * 4; // 4 bytes per pixel (RGBA)

                        byte blue = ptr[index];
                        byte green = ptr[index + 1];
                        byte red = ptr[index + 2];

                        // Calculate the grayscale value (average of RGB values)
                        double averageValue = (red + green + blue) / 3.0;

                        // Add the grayscale value to the sequence
                        pixelSequence.Add(averageValue);
                    }
                }
            }

            // Unlock the image data
            image.UnlockBits(bmpData);

            return pixelSequence;
        }



        private static void PredictNextElement(Predictor predictor, double[] list)
        {
            Console.WriteLine("------------------------------");

            foreach (var item in list)
            {
                var res = predictor.Predict(item);

                if (res.Count > 0)
                {
                    foreach (var pred in res)
                    {
                        Console.WriteLine($"{pred.PredictedInput} - {pred.Similarity}");
                    }

                    var tokens = res.First().PredictedInput.Split('_');
                    var tokens2 = res.First().PredictedInput.Split('-');
                    Console.WriteLine($"Predicted Sequence: {tokens[0]}, predicted next element {tokens2.Last()}");
                }
                else
                    Console.WriteLine("Nothing predicted :(");
            }

            Console.WriteLine("------------------------------");
        }
    }
}
