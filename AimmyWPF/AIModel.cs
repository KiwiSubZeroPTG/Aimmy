using AimmyWPF.Class;
using KdTree;
using KdTree.Math;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace AimmyAimbot
{
    public class AIModel : IDisposable
    {
        private const int IMAGE_SIZE = 640;
        private const int NUM_DETECTIONS = 8400; // Standard for Yolov8 model (Shape: 1x5x8400)

        private readonly RunOptions _modelOptions;
        private InferenceSession _onnxModel;

        public float ConfidenceThreshold = 0.6f; // Adjust confidence level
        public bool CollectData = false; // Toggle data collection for debugging
        public int FovSize = 640; // Field of view size in pixels
        public float AimSmoothing = 0.1f; // Smoothness factor for human-like aim

        private DateTime _lastSavedTime = DateTime.MinValue;
        private List<string> _outputNames;
        private Bitmap _screenCaptureBitmap = null;
        private static readonly byte[] _rgbValuesCache = new byte[640 * 640 * 3];
        private readonly object _bitmapLock = new object();

        public AIModel(string modelPath)
        {
            _modelOptions = new RunOptions();

            var sessionOptions = new SessionOptions
            {
                EnableCpuMemArena = true,
                EnableMemoryPattern = true,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_PARALLEL
            };

            try
            {
                LoadViaDirectML(sessionOptions, modelPath);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"DirectML failed: {ex.Message}\nSwitching to CPU.", "Error");
                LoadViaCPU(sessionOptions, modelPath);
            }

            ValidateOnnxShape();
        }

        private void LoadViaDirectML(SessionOptions sessionOptions, string modelPath)
        {
            sessionOptions.AppendExecutionProvider_DML();
            _onnxModel = new InferenceSession(modelPath, sessionOptions);
            _outputNames = _onnxModel.OutputMetadata.Keys.ToList();
        }

        private void LoadViaCPU(SessionOptions sessionOptions, string modelPath)
        {
            sessionOptions.AppendExecutionProvider_CPU();
            _onnxModel = new InferenceSession(modelPath, sessionOptions);
            _outputNames = _onnxModel.OutputMetadata.Keys.ToList();
        }

        private void ValidateOnnxShape()
        {
            foreach (var output in _onnxModel.OutputMetadata)
            {
                var shape = output.Value.Dimensions;
                if (shape.Length != 3 || shape[0] != 1 || shape[1] != 5 || shape[2] != NUM_DETECTIONS)
                {
                    throw new InvalidOperationException($"Invalid model shape: {string.Join("x", shape)}. Use a Yolov8 ONNX model.");
                }
            }
        }

        public class Prediction
        {
            public RectangleF Rectangle { get; set; }
            public float Confidence { get; set; }
        }

        public static float AIConfidence { get; set; }

        public Bitmap ScreenGrab(Rectangle detectionBox)
        {
            lock (_bitmapLock)
            {
                if (_screenCaptureBitmap == null || _screenCaptureBitmap.Width != detectionBox.Width || _screenCaptureBitmap.Height != detectionBox.Height)
                {
                    _screenCaptureBitmap?.Dispose();
                    _screenCaptureBitmap = new Bitmap(detectionBox.Width, detectionBox.Height);
                }

                using (var g = Graphics.FromImage(_screenCaptureBitmap))
                {
                    g.CopyFromScreen(detectionBox.Left, detectionBox.Top, 0, 0, detectionBox.Size);
                }

                return (Bitmap)_screenCaptureBitmap.Clone();
            }
        }

        public static float[] BitmapToFloatArray(Bitmap image)
        {
            int height = image.Height;
            int width = image.Width;
            float[] result = new float[3 * height * width];
            Rectangle rect = new Rectangle(0, 0, width, height);
            BitmapData bmpData = image.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);

            IntPtr ptr = bmpData.Scan0;
            byte[] rgbValues = new byte[width * height * 3];

            Marshal.Copy(ptr, rgbValues, 0, rgbValues.Length);

            Parallel.For(0, height, y =>
            {
                for (int x = 0; x < width; x++)
                {
                    int index = (y * width + x) * 3;
                    int i = y * width + x;
                    result[i] = rgbValues[index + 2] / 255.0f; // R
                    result[height * width + i] = rgbValues[index + 1] / 255.0f; // G
                    result[2 * height * width + i] = rgbValues[index] / 255.0f; // B
                }
            });

            image.UnlockBits(bmpData);
            return result;
        }

        public async Task<Prediction> GetClosestPredictionToCenterAsync()
        {
            int halfScreenWidth = Screen.PrimaryScreen.Bounds.Width / 2;
            int halfScreenHeight = Screen.PrimaryScreen.Bounds.Height / 2;
            int detectionBoxSize = IMAGE_SIZE;

            Rectangle detectionBox = new Rectangle(halfScreenWidth - detectionBoxSize / 2,
                                                   halfScreenHeight - detectionBoxSize / 2,
                                                   detectionBoxSize,
                                                   detectionBoxSize);

            Bitmap frame = ScreenGrab(detectionBox);

            if (CollectData && (DateTime.Now - _lastSavedTime).TotalSeconds >= 0.5)
            {
                _lastSavedTime = DateTime.Now;
                string uuid = Guid.NewGuid().ToString();
                await Task.Run(() => frame.Save($"bin/images/{uuid}.jpg"));
            }

            float[] inputArray = BitmapToFloatArray(frame);
            if (inputArray == null) return null;

            Tensor<float> inputTensor = new DenseTensor<float>(inputArray, new int[] { 1, 3, frame.Height, frame.Width });
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", inputTensor) };

            using var results = _onnxModel.Run(inputs, _outputNames, _modelOptions);
            var outputTensor = results[0].AsTensor<float>();

            float fovMinX = (IMAGE_SIZE - FovSize) / 2.0f;
            float fovMaxX = (IMAGE_SIZE + FovSize) / 2.0f;
            float fovMinY = (IMAGE_SIZE - FovSize) / 2.0f;
            float fovMaxY = (IMAGE_SIZE + FovSize) / 2.0f;

            var tree = new KdTree<float, Prediction>(2, new FloatMath());

            var filteredIndices = Enumerable.Range(0, NUM_DETECTIONS)
                                            .AsParallel()
                                            .Where(i => outputTensor[0, 4, i] >= ConfidenceThreshold)
                                            .ToList();

            foreach (var i in filteredIndices)
            {
                float objectness = outputTensor[0, 4, i];
                AIConfidence = objectness;

                float x_center = outputTensor[0, 0, i];
                float y_center = outputTensor[0, 1, i];
                float width = outputTensor[0, 2, i];
                float height = outputTensor[0, 3, i];

                float x_min = x_center - width / 2;
                float y_min = y_center - height / 2;
                float x_max = x_center + width / 2;
                float y_max = y_center + height / 2;

                if (x_min >= fovMinX && x_max <= fovMaxX && y_min >= fovMinY && y_max <= fovMaxY)
                {
                    var prediction = new Prediction
                    {
                        Rectangle = new RectangleF(x_min, y_min, x_max - x_min, y_max - y_min),
                        Confidence = objectness
                    };

                    tree.Add(new[] { x_center, y_center }, prediction);
                }
            }

            var nodes = tree.GetNearestNeighbours(new[] { IMAGE_SIZE / 2.0f, IMAGE_SIZE / 2.0f }, 1);

            return nodes.Length > 0 ? SmoothPrediction(nodes[0].Value) : null;
        }

        private Prediction SmoothPrediction(Prediction target)
        {
            // Add smoothing for human-like movement
            float centerX = target.Rectangle.X + target.Rectangle.Width / 2.0f;
            float centerY = target.Rectangle.Y + target.Rectangle.Height / 2.0f;

            centerX = Lerp(Screen.PrimaryScreen.Bounds.Width / 2, centerX, AimSmoothing);
            centerY = Lerp(Screen.PrimaryScreen.Bounds.Height / 2, centerY, AimSmoothing);

            target.Rectangle = new RectangleF(centerX - target.Rectangle.Width / 2,
                                              centerY - target.Rectangle.Height / 2,
                                              target.Rectangle.Width,
                                              target.Rectangle.Height);

            return target;
        }

        private float Lerp(float start, float end, float smoothing)
        {
            return start + (end - start) * smoothing;
        }

        public void Dispose()
        {
            _onnxModel?.Dispose();
            lock (_bitmapLock)
            {
                _screenCaptureBitmap?.Dispose();
            }
            GC.SuppressFinalize(this);
        }
    }
}
