using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DocumentClassificationModels
{
    public class Model
    {
        string projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../"));
        private readonly string _inceptionTensorFlowModel; // путь к модели Inception 
        private MLContext mlContext;
        private ITransformer model;
        private DataViewSchema schema;
        private string modelName = "model.zip"; // название модели для её сохранения
        private string _setsPath = @"D:\test"; // путь к сетам и место, куда будет сложена модель после сохранения
        private PredictionEngine<ImageData, ImagePrediction> predictor;

        public Model(string inceptionTensorFlowModel)
        {
            mlContext = new MLContext();
            _inceptionTensorFlowModel = inceptionTensorFlowModel;
        }
        internal struct InceptionSettings
        {
            public const int ImageHeight = 224;
            public const int ImageWidth = 224;
            public const float Mean = 117;
            public const float Scale = 1;
            public const bool ChannelsLast = true;
        }
        private double TrainModel()
        {
            IEstimator<ITransformer> pipeline = mlContext.Transforms
                .LoadImages(outputColumnName: "input", imageFolder: "", inputColumnName: nameof(ImageData.ImagePath))
                           .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: InceptionSettings.ImageWidth, imageHeight: InceptionSettings.ImageHeight, inputColumnName: "input"))
                           .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: InceptionSettings.ChannelsLast, offsetImage: InceptionSettings.Mean))
                           .Append(mlContext.Model.LoadTensorFlowModel(_inceptionTensorFlowModel).
                               ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true))
                           .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: "Label"))
                           .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "LabelKey", featureColumnName: "softmax2_pre_activation"))
                           .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"))
                           .AppendCacheCheckpoint(mlContext);

            var loadImages = ImageData.ReadData(_setsPath);
            IDataView trainingData = mlContext.Data.LoadFromEnumerable<ImageData>(loadImages.train);
            ITransformer model = pipeline.Fit(trainingData);
            IDataView testData = mlContext.Data.LoadFromEnumerable<ImageData>(loadImages.test);
            IDataView predictions = model.Transform(testData);
            List<ImagePrediction> imagePredictionData = mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, true).ToList();
            MulticlassClassificationMetrics metrics =
                mlContext.MulticlassClassification.Evaluate(predictions,
                  labelColumnName: "LabelKey",
                  predictedLabelColumnName: "PredictedLabel");
            schema = trainingData.Schema;
            mlContext.Model.Save(model, schema, Path.Combine(_setsPath, modelName));
            return metrics.LogLoss;
        }
        public void SaveModel() => mlContext.Model.Save(model, schema, Path.Combine(_setsPath, modelName));
        public void FitModel()
        {
            var LogLoss = TrainModel();
            Console.WriteLine($"LogLoss is {LogLoss}");
            //SaveModel();
        }
        public ImagePrediction ClassifySingleImage(string filePath)
        {
            if (model == null)
                LoadModel();
            if (predictor == null)
                predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
            var imageData = new ImageData()
            {
                ImagePath = filePath
            };
            return predictor.Predict(imageData);
        }
        public void LoadModel() =>
            model = mlContext.Model.Load(Path.Combine(_setsPath, modelName),out var _);

    }
    public class ImageData
    {
        [LoadColumn(0)]
        public string ImagePath;

        [LoadColumn(1)]
        public string Label;

        //метод, который я использую, чтобы забрать данные из папок и отмаркировать их
        public static (IEnumerable<ImageData> train, IEnumerable<ImageData> test) ReadData(string pathToFolder)
        {
            List<ImageData> list = new List<ImageData>();
            var directories = Directory.EnumerateDirectories(pathToFolder);
            foreach (var dir in directories)
            {
                
                var label = dir.Split(@"\").Last();
                foreach (var file in Directory.GetFiles(dir))
                {
                    list.Add(new ImageData()
                    {
                        ImagePath = file,
                        Label = label
                    });
                }
            }
            list = list.Shuffle().ToList();
            return GetSets(list);
        }

        //Делим изображения на тестовую и основную выборки
        public static (IEnumerable<ImageData> train, IEnumerable<ImageData> test) GetSets(IEnumerable<ImageData> data)
        {
            var trainCount = data.Count() / 100 * 99;
            var train = data.Take(trainCount);
            var test = data.Skip(trainCount);
            return (train, test);
        }
    }
    public class ImagePrediction : ImageData
    {
        [ColumnName("Score")]
        public float[] Score;

        public string PredictedLabelValue;

    }
    
}
