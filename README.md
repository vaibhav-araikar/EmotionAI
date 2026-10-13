# Facial Emotion Recognition Using Deep Learning

An end-to-end FER-2013 project using PyTorch, a custom CNN baseline, a ResNet-style model, evaluation metrics, and optional webcam/Streamlit demos. CUDA is used automatically when available.

> This is a starter repository. The FER-2013 dataset and trained checkpoints are not included. Facial expression predictions should not be interpreted as reliable measures of a person's internal emotional state.

## 1. Requirements

- Windows 10/11 (instructions below use PowerShell)
- Python 3.11 recommended
- NVIDIA driver compatible with the installed PyTorch CUDA build
- FER-2013 CSV file with `emotion`, `pixels`, and `Usage` columns

## 2. Create the environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install a CUDA-enabled PyTorch build. Example using CUDA 12.1 wheels (check the official PyTorch install selector if this build is not suitable for your driver):

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Verify GPU availability:

```powershell
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If `torch.cuda.is_available()` is false, troubleshoot PyTorch/driver compatibility before training.

## 3. Add the dataset

Download FER-2013 from Kaggle: https://www.kaggle.com/datasets/msambare/fer2013

Place the CSV at:

```text
dataset/raw/fer2013.csv
```

The loader expects the common CSV format with columns `emotion`, `pixels`, and `Usage`; standard usage values include `Training`, `PublicTest`, and `PrivateTest`.

## 4. Train

Run commands from the repository root:

```powershell
python -m src.training.train --model cnn
```

Train the residual model:

```powershell
python -m src.training.train --model resnet
```

Override epochs or batch size if needed:

```powershell
python -m src.training.train --model cnn --epochs 40 --batch-size 64
```

If CUDA runs out of memory, retry with `--batch-size 32` or `16`. Checkpoints are saved to `checkpoints/`.

## 5. Evaluate

The default test evaluation uses `PrivateTest`. Use it only after model selection is finished; use `PublicTest` for validation during development.

```powershell
python -m src.training.evaluate --model cnn --split PrivateTest
python -m src.training.evaluate --model resnet --split PrivateTest
```

The script prints accuracy, macro-F1, precision, recall, and saves a confusion matrix and report under `results/`.

## 6. Predict one image

```powershell
python -m src.inference.predict path\to\face.jpg --model cnn
```

The image is converted to grayscale and resized to 48x48. For best results, use a reasonably centered, cropped face image; this basic script does not perform face detection on single images.

## 7. Webcam demo

```powershell
python -m src.inference.webcam --model cnn
```

Press `q` to quit. The demo uses OpenCV's Haar cascade for face detection. Lighting, face alignment, camera quality, and differences from FER-2013 can reduce accuracy.

## 8. Optional Streamlit app

```powershell
streamlit run app/app.py
```

Upload a face image and click **Predict expression**. A trained checkpoint must exist first.

## Project structure

```text
dataset/          FER-2013 CSV and optional processed data
notebooks/        Dataset exploration and experiment notebooks
src/data/         Dataset loader and transforms
src/models/       CNN and residual architecture
src/training/     Training, evaluation, metrics
src/inference/    Image prediction and webcam demo
checkpoints/      Saved model weights (not committed)
results/          Reports and confusion matrices
app/              Optional Streamlit interface
config.py         Central training configuration
```

## Suggested experiments

1. Establish a custom CNN baseline.
2. Compare a residual model.
3. Compare augmentation strategies and regularization.
4. Report accuracy and macro-F1; inspect per-class precision/recall and confusion matrix.
5. Keep the test set untouched until final evaluation.
6. Record environment, seed, training duration, and GPU model for reproducibility.

## Known limitations

- FER-2013 is noisy and class-imbalanced; expressions can be ambiguous.
- A model trained on this dataset may generalize poorly to real webcam images.
- Predicted expression categories are not reliable indicators of internal feelings or intent.
