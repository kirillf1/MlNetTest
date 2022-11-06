// See https://aka.ms/new-console-template for more information
using DocumentClassificationModels;
using Microsoft.ML;


var trainer = new DocumentTrainer();
//trainer.Train();
var input = new DocumentInput() { ImagePath = "D:\\test\\UD\\7001-1.jpg" };
var res = trainer.Predict(input);
Console.WriteLine(res.PredictedLabel);
Console.ReadLine();