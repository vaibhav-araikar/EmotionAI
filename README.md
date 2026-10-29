# 😊 EmotionAI — Facial Emotion Recognition Using Deep Learning

A deep learning project that recognizes seven facial expression categories using a custom ResNet-style Convolutional Neural Network (CNN) built with PyTorch.

The project provides a Streamlit web interface with three modes:

- **Live Emotion Detection:** Detect facial expressions continuously through a webcam.
- **Image Upload:** Upload an existing image and predict the facial expression.
- **Photo Capture:** Capture a picture using the camera and analyze the expression.

The application displays a matching emoji, predicted expression, and model confidence score.

> **Important:** This project classifies visible facial expressions. It cannot reliably determine a person's actual internal emotional state. Predictions depend on image quality, lighting, facial visibility, and model accuracy.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Emotion Classes](#-emotion-classes)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [System Requirements](#-system-requirements)
- [GPU vs CPU Requirements](#-gpu-vs-cpu-requirements)
- [Disk Space Requirements](#-disk-space-requirements)
- [Prerequisites and Drivers](#-prerequisites-and-drivers)
- [Clone the Repository](#-clone-the-repository)
- [Installation and Setup](#-installation-and-setup)
- [Dataset Setup](#-dataset-setup)
- [Run the Web Application](#-run-the-web-application)
- [Run Live Webcam Detection](#-run-live-webcam-detection)
- [Train the Model](#-train-the-model)
- [Evaluate the Model](#-evaluate-the-model)
- [Model Performance](#-model-performance)
- [Libraries and Dependencies](#-libraries-and-dependencies)
- [GPU and CPU Configuration](#-gpu-and-cpu-configuration)
- [Troubleshooting](#-troubleshooting)
- [Performance Optimization](#-performance-optimization)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [License](#-license)

---

## 🚀 Project Overview

EmotionAI uses a trained ResNet-style CNN to classify facial images into seven expression categories.

The model accepts a grayscale facial image resized to 48 × 48 pixels and produces a score for each emotion category.

### Workflow

1. Capture a live webcam frame, upload an image, or capture a photo.
2. Detect a face when using live webcam detection.
3. Convert the facial image to grayscale.
4. Resize the input to 48 × 48 pixels.
5. Normalize the image and pass it to the trained neural network.
6. Calculate prediction scores using Softmax.
7. Display the predicted expression, corresponding emoji, and confidence score.

### Model Architecture

- Model: ResNet-style CNN
- Framework: PyTorch
- Input: 48 × 48 grayscale image
- Output: Seven emotion classes
- Checkpoint: `checkpoints/resnet_best.pth`

---

## ⚡ Quick Start — How to Run EmotionAI

Follow these steps after cloning the repository. Run commands from the project root folder.

### 1. Clone the repository

```powershell
git clone https://github.com/vaibhav-araikar/EmotionAI.git
cd EmotionAI
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this in the current terminal and try activation again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install PyTorch

Use the official PyTorch selector to choose a compatible CPU build or CUDA build for your NVIDIA GPU: https://pytorch.org/get-started/locally/

Verify it:

```powershell
python -c "import torch, torchvision; print('PyTorch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### 4. Install project dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If installation fails on PyArrow, see [PyArrow Installation Fails](#2-pyarrow-installation-fails) in Troubleshooting. Python and package versions must be compatible; don't force an incompatible PyArrow version.

### 5. Check the trained model checkpoint

The application expects this file:

```text
checkpoints/resnet_best.pth
```

If it is not included in the repository, obtain the checkpoint from the project's release/download location or train the model using the dataset instructions below. The app cannot predict without a compatible checkpoint.

### 6. Run the Streamlit web application

From the project root, with the virtual environment activated:

```powershell
python -m streamlit run app/app.py
```

Open the URL printed in the terminal, usually http://localhost:8501.

- **🎥 Live emotion detection:** Click **START** and allow browser camera access.
- **📁 Upload an image:** Select a JPG, JPEG, or PNG file.
- **📷 Capture a photo:** Allow camera access and capture a picture.

### 7. Run the standalone webcam app (optional)

```powershell
python -m src.inference.webcam
```

### 8. Train or evaluate the model (optional)

Training requires the dataset folder structure described in [Dataset Setup](#-dataset-setup).

Train ResNet:

```powershell
python -m src.training.train --model resnet --epochs 30 --batch-size 32
```

Evaluate ResNet:

```powershell
python -m src.training.evaluate --model resnet
```

To stop Streamlit, return to the terminal and press `Ctrl + C`.

---

## ✨ Features

- Real-time webcam facial expression detection
- Image upload and expression prediction
- Photo capture and analysis
- Seven emotion categories
- Emotion-specific emojis
- Confidence score visualization
- Scores for all seven classes
- GPU acceleration when a compatible CUDA-enabled PyTorch installation is available
- CPU inference support
- Streamlit-based interactive user interface
- Model training and test evaluation scripts
- Confusion matrix and classification report generation

---

## 😀 Emotion Classes

| Class      | Display Name | Emoji |
| ---------- | ------------ | ----- |
| `angry`    | Angry        | 😠    |
| `disgust`  | Disgust      | 🤢    |
| `fear`     | Fear         | 😨    |
| `happy`    | Happy        | 😄    |
| `neutral`  | Neutral      | 😐    |
| `sad`      | Sad          | 😢    |
| `surprise` | Surprise     | 😲    |

The emoji corresponds to the predicted class. The confidence score is the model's output score, not a guarantee that the prediction is correct.

---

## 🛠 Technology Stack

| Technology       | Purpose                                   |
| ---------------- | ----------------------------------------- |
| Python           | Main programming language                 |
| PyTorch          | Deep learning and model inference         |
| Torchvision      | Image transformations and model utilities |
| OpenCV           | Face detection and webcam processing      |
| Streamlit        | Web application interface                 |
| Streamlit-WebRTC | Live browser-based camera streaming       |
| PyAV (`av`)      | Video frame processing                    |
| Pillow           | Image loading and processing              |
| NumPy            | Numerical operations                      |
| Scikit-learn     | Evaluation metrics and data splitting     |
| Pandas           | Data and report processing                |
| Matplotlib       | Confusion matrix visualization            |
| Seaborn          | Visualization support                     |
| tqdm             | Training progress bars                    |

---

## 📁 Project Structure

```text
Facial-Emotion-Recognition/
│
├── app/
│   └── app.py
│
├── checkpoints/
│   └── resnet_best.pth
│
├── config.py
│
├── dataset/
│   └── raw/
│       ├── train/
│       │   ├── angry/
│       │   ├── disgust/
│       │   ├── fear/
│       │   ├── happy/
│       │   ├── neutral/
│       │   ├── sad/
│       │   └── surprise/
│       │
│       └── test/
│           ├── angry/
│           ├── disgust/
│           ├── fear/
│           ├── happy/
│           ├── neutral/
│           ├── sad/
│           └── surprise/
│
├── results/
│
├── src/
│   ├── models/
│   │   ├── cnn.py
│   │   └── resnet.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   └── evaluate.py
│   │
│   └── inference/
│       └── webcam.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

**Note:** Generated checkpoints, the dataset, and result files may be excluded from the GitHub repository to keep it small. If they are not included, obtain the dataset and train the model or download the appropriate checkpoint separately before running the relevant features.

---

## 💻 System Requirements

The following are practical estimates, not strict hardware benchmarks.

| Component        | Minimum                              | Recommended                                   |
| ---------------- | ------------------------------------ | --------------------------------------------- |
| Operating system | Windows 10 64-bit                    | Windows 11 64-bit                             |
| Processor        | Modern dual-core CPU                 | Intel Core i5 / AMD Ryzen 5 or equivalent     |
| RAM              | 8 GB                                 | 16 GB or more                                 |
| Python           | Compatible Python 3.x                | Python 3.12 for a reproducible setup          |
| GPU              | Not required                         | NVIDIA GPU with 4 GB or more VRAM             |
| Storage          | 5 GB free for a complete environment | 10–15 GB free, excluding large dataset copies |
| Camera           | Required for live and capture modes  | 720p webcam or better                         |
| Internet         | Required for initial installation    | Stable broadband connection                   |

A dedicated GPU is **optional**. The project can run on CPU-only laptops, although live detection and training may be slower.

### Python Version Compatibility

Python package compatibility depends on the versions of PyTorch, Streamlit, PyArrow, OpenCV, and their dependencies.

Python 3.12 is recommended for a reproducible installation. Newer Python versions, including Python 3.15, may work with suitable wheels, but some dependencies may not yet support them consistently.

For Python 3.15, check package compatibility and use the troubleshooting instructions below if installation fails.

---

## 🎮 GPU vs CPU Requirements

### Option A: NVIDIA GPU Laptop

Recommended for faster model training and smoother live inference.

Requirements:

- NVIDIA GPU
- Compatible NVIDIA display driver
- CUDA-enabled PyTorch and Torchvision builds
- Approximately 4 GB or more dedicated VRAM for comfortable experimentation
- 16 GB system RAM recommended

Check the NVIDIA driver:

```powershell
nvidia-smi
```

Check PyTorch GPU availability:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode')"
```

If CUDA is available, the application can select the GPU automatically.

**Driver note:** The NVIDIA driver must support the CUDA runtime bundled with your PyTorch installation. You normally do not need to install the full CUDA Toolkit separately when using compatible PyTorch pip binaries.

### Option B: CPU-Only Laptop

A dedicated GPU is not necessary.

Requirements:

- 8 GB RAM minimum; 16 GB recommended
- Modern multi-core processor
- Compatible CPU build of PyTorch and Torchvision
- Sufficient free disk space

The application automatically falls back to CPU if CUDA is unavailable.

CPU inference may have a lower frame rate, especially when face detection and neural network inference are performed on every frame. Use a smaller frame size and less frequent predictions if needed.

---

## 💾 Disk Space Requirements

Storage usage depends on the operating system, package versions, dataset, cache files, and whether GPU-enabled PyTorch packages are installed.

The following estimates are approximate planning ranges.

| Component                                      |                                  Approximate disk space |
| ---------------------------------------------- | ------------------------------------------------------: |
| Git repository source code                     |                                                 1–20 MB |
| Python virtual environment basics              |                                               50–200 MB |
| PyTorch and Torchvision                        |                                                  1–5 GB |
| CUDA-related libraries bundled with GPU wheels | Included in the GPU package total; varies significantly |
| OpenCV                                         |                                               50–300 MB |
| Streamlit, WebRTC, and related packages        |                                              100–500 MB |
| Pillow, NumPy, Pandas, and other utilities     |                                              100–700 MB |
| Dataset                                        |                                     100 MB–2 GB or more |
| Trained checkpoints                            |   A few MB to hundreds of MB, depending on architecture |
| Training outputs and caches                    |                                       100 MB–several GB |

These ranges overlap because packages share dependencies. Do not add every upper bound and treat it as an exact requirement.

### Practical Storage Planning

| Setup                                         | Suggested free space |
| --------------------------------------------- | -------------------: |
| Source code only                              |               100 MB |
| CPU inference environment                     |               3–6 GB |
| GPU inference environment                     |              5–10 GB |
| Training environment with dataset and results |     10–15 GB or more |

There is no fixed maximum storage requirement: dataset size, saved checkpoints, package caches, and repeated experiments can increase usage.

For this project's approximately 28,709 training images and 6,965 test images, reserve additional space for the dataset and any extracted copies.

---

## 📦 Prerequisites and Drivers

Install the following before starting:

1. Git: https://git-scm.com/downloads
2. Python: https://www.python.org/downloads/
3. A code editor such as Visual Studio Code: https://code.visualstudio.com/
4. NVIDIA driver, if using an NVIDIA GPU: https://www.nvidia.com/Download/index.aspx

Verify the installations:

```powershell
git --version
python --version
python -m pip --version
```

If `python` is not recognized, reinstall Python and enable **Add Python to PATH**, or use the correct Python launcher command for your system.

---

## 📥 Clone the Repository

Open PowerShell or the VS Code terminal.

Replace the repository URL below with the actual URL of your GitHub repository.

```powershell
git clone https://github.com/vaibhav-araikar/EmotionAI.git
cd EmotionAI
```

If the project is already cloned:

```powershell
cd Facial-Emotion-Recognition
git pull
```

If Git is unavailable, open your GitHub repository, select **Code → Download ZIP**, extract it, and open the extracted folder in VS Code.

---

## 🐍 Installation and Setup

### Step 1: Create a Virtual Environment

```powershell
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip --version
```

### Step 2: Upgrade Installation Tools

```powershell
python -m pip install --upgrade pip setuptools wheel
```

### Step 3: Install PyTorch

For CPU-only inference, install a compatible CPU build using the official PyTorch selector:

https://pytorch.org/get-started/locally/

For NVIDIA GPU acceleration, choose the CUDA option supported by your operating system and GPU driver on the same page.

After installation, verify the setup:

```powershell
python -c "import torch; import torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.cuda.is_available())"
```

### Step 4: Install Project Dependencies

Create a `requirements.txt` containing the compatible versions used by your project. At minimum, the application requires:

```text
streamlit
streamlit-webrtc
av
torch
torchvision
opencv-python
Pillow
numpy
pandas
scikit-learn
matplotlib
seaborn
tqdm
```

Install the requirements:

```powershell
python -m pip install -r requirements.txt
```

**Important:** The exact package versions should be pinned and tested together before publishing a reproducible release. In particular, Streamlit's PyArrow requirement can conflict with some newer Python/package combinations. Installing `streamlit` alone may fail if pip cannot find a compatible PyArrow wheel.

If your working environment uses a custom dependency workaround, document that separately rather than assuming it works for every installation.

### Step 5: Verify the Important Libraries

```powershell
python -c "import torch, torchvision, cv2, streamlit, av; from PIL import Image; print('All core imports successful'); print('CUDA available:', torch.cuda.is_available()); print('OpenCV:', cv2.__version__); print('Streamlit:', streamlit.__version__); print('AV:', av.__version__)"
```

---

## 🗂 Dataset Setup

The training script expects the dataset organized into class-specific directories.

```text
dataset/raw/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
│
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

Place the appropriate image files inside their corresponding directories.

The current dataset contains approximately:

- Training images: 28,709
- Test images: 6,965
- Total images: 35,674
- Emotion categories: 7

These are the dataset counts used during development. If you use a different dataset, counts and results will differ.

**Dataset note:** Confirm the dataset's license and redistribution requirements before publishing the images or including them in a public GitHub repository.

---

## 🌐 Run the Web Application

From the project root, with the virtual environment activated:

```powershell
python -m streamlit run app/app.py
```

Open the local URL shown in your terminal. Usually:

```text
http://localhost:8501
```

### Mode 1: Live Emotion Detection

1. Open the **Live emotion detection** tab.
2. Click **START**.
3. Allow browser camera access.
4. Position your face inside the frame.
5. Keep your face visible and use good lighting.
6. Observe the expression label, emoji, and confidence score.

The live mode uses WebRTC to stream camera frames to the application. Camera access generally requires localhost or HTTPS.

### Mode 2: Upload an Image

1. Open the **Upload an image** tab.
2. Select a JPG, JPEG, or PNG image.
3. Wait for prediction.
4. Review the predicted expression and scores.

### Mode 3: Capture a Photo

1. Open the **Capture a photo** tab.
2. Allow camera access.
3. Capture an image.
4. Review the predicted expression and score breakdown.

### Stop the Application

Return to the terminal and press:

```text
Ctrl + C
```

---

## 🎥 Run Live Webcam Detection Separately

The project also includes a standalone OpenCV webcam inference script.

From the project root:

```powershell
python -m src.inference.webcam
```

This script uses the trained checkpoint and OpenCV's Haar cascade face detector.

To stop the webcam window, use the exit key implemented in the script, commonly `q`.

---

## 🧠 Train the Model

Ensure the dataset is correctly organized before training.

### Train the ResNet Model

```powershell
python -m src.training.train --model resnet --epochs 30 --batch-size 32
```

### Train the CNN Model

```powershell
python -m src.training.train --model cnn --epochs 30 --batch-size 64
```

The training pipeline uses the folder-based dataset, creates a training/validation split, tracks validation performance, and saves the best checkpoint.

Training time depends on the processor, GPU, batch size, number of epochs, and dataset.

**Warning:** Training may overwrite an existing checkpoint with the same output filename. Back up important checkpoints before retraining.

---

## 📊 Evaluate the Model

Evaluate the ResNet model:

```powershell
python -m src.training.evaluate --model resnet
```

Evaluate the CNN model:

```powershell
python -m src.training.evaluate --model cnn
```

The evaluation script uses the test dataset and saves evaluation outputs under the project's results directory, according to the configured paths.

Typical evaluation outputs include a classification report and confusion matrix.

---

## 📈 Model Performance

The following results were obtained during development on the project's test dataset.

| Metric            |    CNN | ResNet |
| ----------------- | -----: | -----: |
| Test accuracy     | 63.62% | 65.48% |
| Macro F1-score    | 0.5963 | 0.6310 |
| Weighted F1-score | 0.6298 | 0.6515 |

ResNet performed better than the CNN in these experiments, although neither model is perfectly accurate.

Performance may differ when the model is tested on different people, lighting conditions, camera hardware, or datasets.

---

## 📚 Libraries and Dependencies

| Library            | Purpose                                   |
| ------------------ | ----------------------------------------- |
| `torch`            | Model inference and training              |
| `torchvision`      | Image preprocessing                       |
| `streamlit`        | Interactive web interface                 |
| `streamlit-webrtc` | Browser webcam streaming                  |
| `av`               | Video frame conversion                    |
| `opencv-python`    | Face detection and image operations       |
| `Pillow`           | Image loading, resizing, and conversion   |
| `numpy`            | Array operations                          |
| `pandas`           | Data processing                           |
| `scikit-learn`     | Classification metrics and data splitting |
| `matplotlib`       | Plotting                                  |
| `seaborn`          | Confusion matrix visualization            |
| `tqdm`             | Progress indicators                       |

Additional packages are installed automatically as dependencies.

---

## 🖥 GPU and CPU Configuration

The application selects its inference device automatically:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

### Verify GPU Detection

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

If the output is `True`, check the GPU name:

```powershell
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

If the output is `False`, the application should use the CPU. This does not necessarily mean your laptop is broken; your PyTorch installation may be CPU-only or CUDA may not be configured correctly.

---

## 🛠 Troubleshooting

### 1. `ModuleNotFoundError: No module named 'streamlit'`

Activate the correct virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then install the package:

```powershell
python -m pip install streamlit
```

Run the app through the same Python interpreter:

```powershell
python -m streamlit run app/app.py
```

### 2. PyArrow Installation Fails

Some Python versions may not have compatible PyArrow wheels for the version required by Streamlit.

Check the available Python version:

```powershell
python --version
```

Check available PyArrow versions:

```powershell
python -m pip index versions pyarrow
```

Check whether a compatible wheel exists:

```powershell
python -m pip install --only-binary=:all: "pyarrow<26"
```

If pip reports that no compatible distribution exists, do not repeatedly attempt to compile PyArrow. Use a Python version supported by the required Streamlit/PyArrow combination, or select compatible package versions after verifying their dependency constraints.

Installing a different PyArrow version without checking Streamlit's requirements can cause dependency conflicts.

### 3. `ModuleNotFoundError: No module named 'torch'`

Install PyTorch in the active virtual environment:

```powershell
python -m pip install torch torchvision
```

For a specific CUDA configuration, use the official PyTorch installation selector.

### 4. CUDA Is Not Available

Check the driver:

```powershell
nvidia-smi
```

Check PyTorch:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Possible causes:

- CPU-only PyTorch was installed.
- NVIDIA drivers are missing or outdated.
- The driver is incompatible with the selected CUDA runtime.
- PyTorch and Torchvision versions are incompatible.

Install a compatible PyTorch and Torchvision combination rather than randomly changing CUDA packages.

### 5. `Checkpoint not found`

Ensure this file exists:

```text
checkpoints/resnet_best.pth
```

If the checkpoint was not included in the GitHub repository, train the model or obtain the checkpoint from the project's official release or download location.

Run Streamlit from the project root:

```powershell
python -m streamlit run app/app.py
```

### 6. `The checkpoint is missing class_names metadata`

The application expects checkpoint metadata containing the class names.

Use a checkpoint produced by the project's training pipeline, or update the loading logic carefully to match the exact output-class order used during training.

The class order must match the model's training order.

### 7. Webcam Does Not Open

Check:

- Camera permissions in your browser.
- Whether another application is using the camera.
- Windows camera privacy settings.
- Whether the camera works in the operating system's Camera application.
- Whether the app is running on localhost or HTTPS.

Restart the browser and Streamlit after changing permissions.

### 8. Live Video Freezes or Stops

Possible causes include excessive inference workload, insufficient CPU/GPU resources, browser limitations, or WebRTC processing errors.

Try:

- Closing other applications using the webcam.
- Improving lighting so the face detector works reliably.
- Lowering the video resolution.
- Running fewer predictions per second.
- Checking the Streamlit terminal for a traceback.
- Restarting the application.

Do not silently ignore exceptions while debugging. Print or log the full exception so the underlying problem can be identified.

### 9. `cv2.CascadeClassifier` Is Missing

Check the installed OpenCV package:

```powershell
python -c "import cv2; print(cv2.__version__); print(hasattr(cv2, 'CascadeClassifier')); print(cv2.data.haarcascades)"
```

Use the standard OpenCV package:

```powershell
python -m pip uninstall opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless
python -m pip install opencv-python
```

Avoid installing multiple conflicting OpenCV wheel variants in the same environment.

### 10. `ImportError` or Version Conflicts

Inspect the environment:

```powershell
python -m pip check
```

Inspect important package versions:

```powershell
python -m pip show torch torchvision streamlit streamlit-webrtc av opencv-python
```

Use tested compatible versions. Avoid upgrading every package at once, because that can introduce new incompatibilities.

### 11. Port 8501 Is Already in Use

Start Streamlit on another port:

```powershell
python -m streamlit run app/app.py --server.port 8502
```

Open:

```text
http://localhost:8502
```

### 12. PowerShell Does Not Allow Virtual Environment Activation

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

This changes the execution policy only for the current PowerShell process.

Alternatively, use the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

### 13. Model Predictions Are Inaccurate

Try:

- Using a clear, front-facing image.
- Ensuring sufficient lighting.
- Avoiding heavy occlusion of the face.
- Checking that the correct checkpoint is loaded.
- Ensuring preprocessing matches training.
- Evaluating the model using the test script.

A higher confidence score does not guarantee a correct prediction.

---

## ⚡ Performance Optimization

For smoother live detection:

- Use an NVIDIA GPU with compatible CUDA-enabled PyTorch when available.
- Reduce video resolution if the stream is unstable.
- Avoid running multiple training jobs while using live inference.
- Close unnecessary browser tabs and applications.
- Keep the face centered and well-lit.
- Avoid repeated model loading during each video frame.
- Reuse the loaded model and inference resources.

CPU-only laptops can run the application, but training and real-time video inference may be slower.

---

## ⚠️ Limitations

- The model predicts one of seven trained categories.
- Predictions depend on dataset quality and representation.
- Facial expressions are not definitive indicators of internal emotional states.
- Performance can decrease under poor lighting, unusual poses, occlusion, or camera blur.
- Webcam performance depends on hardware, browser permissions, and streaming stability.
- The reported test accuracy is specific to the evaluation dataset and does not guarantee real-world accuracy.

---

## 🔮 Future Improvements

- Improve model accuracy and generalization.
- Add a more robust face detector.
- Support multiple faces simultaneously.
- Smooth predictions across video frames.
- Add prediction history and visualization.
- Optimize CPU inference for low-end laptops.
- Export a deployment-friendly model format.
- Add automated tests and a pinned dependency file.
- Provide a downloadable, versioned model checkpoint.

---

## 🤝 Contributing

Contributions, bug reports, and suggestions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Test the application.
5. Submit a pull request.

Please include reproduction steps and relevant error messages when reporting bugs.

---

## 📄 License

Add a `LICENSE` file before distributing this project publicly. Choose a license appropriate for your code and verify the licensing terms of the dataset and any pretrained assets you use.

---

## 👨‍💻 Acknowledgements

Built using Python, PyTorch, Torchvision, OpenCV, Streamlit, and Streamlit-WebRTC.

**EmotionAI — Facial Expression Recognition Using Deep Learning**

<!-- just completing my streak -->
<!-- and not losing my streaks  -->
