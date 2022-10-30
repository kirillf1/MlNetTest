using Microsoft.Win32;
using MlNetTest_UI;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace MlNetTest.UI
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new OpenFileDialog()
            {
                Title = "Выбор картинки",
                CheckFileExists = true,
            };
            if (dialog.ShowDialog(this) == false)
                return;
            var file = dialog.FileName;
            this.ImageView.Source = new BitmapImage(new Uri(file));
            var analyzeData = new DocumentClassification.ModelInput()
            {
                ImageSource = file,
            };
            var result =  DocumentClassification.Predict(analyzeData);
            Result.Text = $"Это: {result.Prediction} верятность: {result.Score.Max() :p0}";
        }
    }
}
