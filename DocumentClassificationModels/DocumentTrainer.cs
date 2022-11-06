using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using Microsoft.ML.Vision;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using Tensorflow.Contexts;
using static Microsoft.ML.DataOperationsCatalog;

namespace DocumentClassificationModels
{
    public class DocumentTrainer
    {
        public void Train()
        {
            var projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../"));
            var workspaceRelativePath = Path.Combine(projectDirectory, "workspace");
            var assetsRelativePath = @"D:\test";
            MLContext mlContext = new MLContext();
            var images = LoadImagesFromDirectory(folder: assetsRelativePath, useFolderNameAsLabel: true);
            IDataView imageData = mlContext.Data.LoadFromEnumerable(images);
            IDataView shuffledData = mlContext.Data.ShuffleRows(imageData);
            //var resizing = mlContext.Transforms
            //    .ResizeImages("input", 300, 300, "input");
            var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Label", outputColumnName: "LabelAsKey")
    .Append(mlContext.Transforms.LoadRawImageBytes(
        outputColumnName: "Image",
        imageFolder: assetsRelativePath,
        inputColumnName: "ImagePath"));
            IDataView preProcessedData = preprocessingPipeline
                    .Fit(shuffledData)
                    .Transform(shuffledData);
            TrainTestData trainSplit = mlContext.Data.TrainTestSplit(data: preProcessedData, testFraction: 0.3);
            TrainTestData validationTestSplit = mlContext.Data.TrainTestSplit(trainSplit.TestSet);
            IDataView trainSet = trainSplit.TrainSet;
            IDataView validationSet = validationTestSplit.TrainSet;
            IDataView testSet = validationTestSplit.TestSet;
            var classifierOptions = new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = "Image",
                LabelColumnName = "LabelAsKey",
                ScoreColumnName = "Score",
                ValidationSet = validationSet,
                Epoch = 400,
                Arch = ImageClassificationTrainer.Architecture.ResnetV250,
                MetricsCallback = (metrics) => Console.WriteLine(metrics),
                TestOnTrainSet = true,
                ReuseTrainSetBottleneckCachedValues = true,
                ReuseValidationSetBottleneckCachedValues = true
            };
            var trainingPipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(classifierOptions)
    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            ITransformer trainedModel = trainingPipeline.Fit(trainSet);
            ClassifySingleImage(mlContext, testSet, trainedModel);
            ClassifyImages(mlContext, testSet, trainedModel);
            mlContext.Model.Save(trainedModel, trainSet.Schema, Path.Combine(projectDirectory, "workspace", "model.zip"));
        }
        
     public void ClassifyImages(MLContext mlContext, IDataView data, ITransformer trainedModel)
        {
            IDataView predictionData = trainedModel.Transform(data);
            IEnumerable<DocumentOutput> predictions = mlContext.Data
                .CreateEnumerable<DocumentOutput>(predictionData, reuseRowObject: true)
                .Take(10);
            Console.WriteLine("Classifying multiple images");
            foreach (var prediction in predictions)
            {
                OutputPrediction(prediction);
            }
        }
        public DocumentOutput Predict(DocumentInput document)
        {
            var mlContext = new MLContext();
            var projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../"));
            var workspaceRelativePath = Path.Combine(projectDirectory, "workspace");
            var trainedModel = mlContext.Model.Load(Path.Combine(projectDirectory, "workspace", "model.zip"), out var _);
            PredictionEngine<DocumentInput, DocumentOutput> predictionEngine =
              mlContext.Model.CreatePredictionEngine<DocumentInput, DocumentOutput>(trainedModel);
           var modelOutput =  predictionEngine.Predict(document);
            var labelBuffer = new VBuffer<ReadOnlyMemory<char>>();
            predictionEngine.OutputSchema["Score"].Annotations.GetValue("SlotNames", ref labelBuffer);
            var labels = labelBuffer.DenseValues().Select(l => l.ToString()).ToArray();
            var index = Array.IndexOf(labels, modelOutput.PredictedLabel);
            var score = modelOutput.Score[index];
            return modelOutput;
        }
        private static void OutputPrediction(DocumentOutput prediction)
        {
            string imageName = Path.GetFileName(prediction.ImagePath);
            Console.WriteLine($"Image: {imageName} | Actual Value: {prediction.Label} | Predicted Value: {prediction.PredictedLabel}");
        }
        void ClassifySingleImage(MLContext mlContext, IDataView data, ITransformer trainedModel)
        {
            PredictionEngine<DocumentInput, DocumentOutput> predictionEngine =
                mlContext.Model.CreatePredictionEngine<DocumentInput, DocumentOutput>(trainedModel);
            var image = mlContext.Data.CreateEnumerable<DocumentInput>(data, reuseRowObject: true).First();
            var prediction = predictionEngine.Predict(image);
            Console.WriteLine("Classifying single image");
            OutputPrediction(prediction);
        }
        IEnumerable<DocumentData> LoadImagesFromDirectory(string folder, bool useFolderNameAsLabel = true)
        {
            var files = Directory.GetFiles(folder, "*", searchOption: SearchOption.AllDirectories);
            foreach (var file in files)
            {
                if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png"))
                    continue;
                var label = Path.GetFileName(file);

                if (useFolderNameAsLabel)
                    label = Directory.GetParent(file)!.Name;
                else
                {
                    for (int index = 0; index < label.Length; index++)
                    {
                        if (!char.IsLetter(label[index]))
                        {
                            label = label.Substring(0, index);
                            break;
                        }
                    }
                }
                yield return new DocumentData(file, label);
            }

        }
        //        public void Train(string documentPath)
        //        {
        //            var mlContext = new MLContext();
        //            var data =  mlContext.Data.LoadFromTextFile<DocumentInput>(documentPath);
        //            var converter = mlContext.Transforms.Conversion.MapValueToKey("LabelKey", "Label");
        //            var loading = mlContext.Transforms.LoadImages("input", _trainImagesFolder,"ImagePath");
        //            var resizing = mlContext.Transforms
        //                .ResizeImages("input", InceptionSettings.ImageWidth,InceptionSettings.ImageHeight,"input");
        //            var extracting = mlContext.Transforms.ExtractPixels("input",
        //             null, ImagePixelExtractingEstimator.ColorBits.Alpha, ImagePixelExtractingEstimator.ColorsOrder.ARGB);
        //            var inceptionPipeline = mlContext
        // .Model.LoadTensorFlowModel(tfModelPath)
        // .ScoreTensorFlowModel(new[] {
        //"softmax2_pre_activation" }, new[] { "input" },
        //true);


    }
}

