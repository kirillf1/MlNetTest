// See https://aka.ms/new-console-template for more information
using DocumentClassificationModels;
using Microsoft.ML;


//var trainer = new DocumentTrainer();
//trainer.Train();
//var input = new DocumentInput() { ImagePath = "D:\\test\\CD\\7001-21.jpg" };
//var res = trainer.Predict(input);
//Console.WriteLine(res.PredictedLabel);


//var trainer = new NewTrain();
//trainer.Train();
//Console.ReadLine();



Model model = new Model(@"C:\tensorflow_inception_graph.pb");
model.FitModel();
var res = model.ClassifySingleImage("D:\\test\\CD\\7001-21.jpg");
//model.FitModel();
Console.ReadLine();