using Microsoft.ML.Data;

namespace DocumentClassificationModels
{
    public class DocumentData
    {
       
        public string ImagePath { get; set; } = default!;
        public string Label { get; set; } = default!;
    }
}
