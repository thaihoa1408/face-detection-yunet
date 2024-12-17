using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;
using System;
using System.Drawing;
using System.Windows.Forms;

class Program
{
    [STAThread]
    static void Main()
    {
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        Application.Run(new FaceDetectionForm());
    }
}

public class FaceDetectionForm : Form
{
    private RadioButton useWebcamRadio;
    private RadioButton useFileRadio;
    private Button startButton;
    private OpenFileDialog openFileDialog;
    private string? videoFilePath;

    public FaceDetectionForm()
    {
        Text = "Face Detection";
        Size = new Size(400, 200);

        useWebcamRadio = new RadioButton
        {
            Text = "Use Webcam",
            Location = new Point(20, 20),
            Checked = true
        };
        useFileRadio = new RadioButton
        {
            Text = "Use Video File",
            Location = new Point(20, 50)
        };
        startButton = new Button
        {
            Text = "Start",
            Location = new Point(20, 100)
        };

        startButton.Click += StartButton_Click;

        Controls.Add(useWebcamRadio);
        Controls.Add(useFileRadio);
        Controls.Add(startButton);

        openFileDialog = new OpenFileDialog
        {
            Filter = "Video Files|*.mp4;*.avi;*.mkv;*.mov|All Files|*.*",
            Title = "Select a Video File"
        };
    }

    private void StartButton_Click(object? sender, EventArgs e)
    {
        if (useFileRadio.Checked)
        {
            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                videoFilePath = openFileDialog.FileName;
            }
            else
            {
                MessageBox.Show("No video file selected.", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
        }

        var useWebcam = useWebcamRadio.Checked;
        RunFaceDetection(useWebcam, videoFilePath);
    }

    private void RunFaceDetection(bool useWebcam, string? pathToVideo)
    {
        var windowName = "Face Detection (Press any key to close)";
        CvInvoke.NamedWindow(windowName);

        using var capture = useWebcam
            ? new VideoCapture(camIndex: 0)
            : new VideoCapture(fileName: pathToVideo);

        using var model = InitializeFaceDetectionModel(new Size(capture.Width, capture.Height));

        while (CvInvoke.WaitKey(1) == -1)
        {
            var frame = capture.QueryFrame();
            if (frame is null)
            {
                break;
            }

            var faces = new Mat();
            model.Detect(frame, faces);
            DrawDetectedFaces(frame, faces);

            CvInvoke.Imshow(windowName, frame);
        }

        // Destroy window on close
        CvInvoke.DestroyAllWindows();

        FaceDetectorYN InitializeFaceDetectionModel(Size inputSize) => new FaceDetectorYN(
            model: "face_detection_yunet_2022mar.onnx",
            config: string.Empty,
            inputSize: inputSize,
            scoreThreshold: 0.9f,
            nmsThreshold: 0.3f,
            topK: 5000,
            backendId: Emgu.CV.Dnn.Backend.Default,
            targetId: Target.Cpu);

        void DrawDetectedFaces(Mat frame, Mat faces)
        {
            if (faces.Rows <= 0)
            {
                return;
            }

            var facesData = (float[,])faces.GetData(jagged: true);

            for (var i = 0; i < facesData.GetLength(0); i++)
            {
                var x = (int)facesData[i, 0];
                var y = (int)facesData[i, 1];
                var width = (int)facesData[i, 2];
                var height = (int)facesData[i, 3];

                var faceRectangle = new Rectangle(x, y, width, height);
                CvInvoke.Rectangle(frame, faceRectangle, new MCvScalar(0, 255, 0), 1);

                // Draw confidence
                var confidence = facesData[i, 14];
                CvInvoke.PutText(frame, $"{confidence:N4}", new Point(x, y - 5), FontFace.HersheyComplex, 0.3, new MCvScalar(0, 0, 255), 1);
            }
        }
    }
}
