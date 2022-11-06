using Microsoft.ML.Data;

namespace DocumentClassificationModels
{
    public class DocumentData
    {
        public DocumentData(string imagePath, string label)
        {
            ImagePath = imagePath;
            Label = label;
        }
        public string ImagePath { get; set; } = default!;
        public string Label { get; set; } = default!;
    }
}
