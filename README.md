# Deepfake Detection via Region-Based Temporal Attention

This repository contains the implementation of a deepfake detection pipeline. The project investigates the "Feature Dilution" hypothesis: that standard full-frame resizing erases critical high-frequency artifacts found in the mouth and eye regions.

By comparing **Control models** (Global Pooling) against **Attention models** across different facial regions (Full Face, Eyes, Mouth), this study demonstrates that high-resolution cropping combined with temporal attention significantly outperforms full-frame baselines.

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| **`preprocess.py`** | **The Data Engine.** Converts raw video files (`.mp4`) into optimized NumPy arrays (`.npy`). It performs face detection, extracts 3 regions (Full, Eyes, Mouth) in a single pass, resizes to 224x224 using high-quality interpolation, and saves the data for fast loading. |
| **`models.py`** | **The Architectures.** Contains the TensorFlow/Keras model definitions. Includes the shared backbone (TimeDistributed EfficientNetB1 + BiLSTM) and the two aggregation heads: **Control** (Global Average Pooling) and **Attention** (Temporal Soft Attention). |
| **`run.py`** | **The Orchestrator.** Manages the training pipeline. It handles GPU memory safety, initializes the `VideoDataGenerator` (which loads `.npy` files), performs training with early stopping, and generates the final Confusion Matrix evaluation. |
| **`haarcascade...`** | The OpenCV XML file required for face detection logic in `preprocess.py`. |

## 🧪 Experiment Design

### 1. Dataset & Scope
* **Source:** [FaceForensics++ (C23)](https://github.com/ondyari/FaceForensics)
* **Subset:** `original` (Real) vs. `Face2Face` (Fake).
* **Why Face2Face?** Unlike face-swapping, Face2Face preserves the source identity but manipulates expressions (mouth/eyes). This makes it the ideal candidate for testing temporal jitter detection.
* **Scale:** 1,000 Real Videos / 1,000 Fake Videos.

### 2. Preprocessing Pipeline
To eliminate I/O bottlenecks during training, raw videos are pre-processed into static binary files:
* **Sampling:** 32 frames per video, linearly spaced to capture the full temporal range.
* **Resolution:** 224x224 pixels (upgraded from 96x96 to capture texture artifacts).
* **Regions:**
    1. **Full Face:** Standard baseline.
    2. **Eyes:** Top 35% of the face crop.
    3. **Mouth:** Bottom 35% of the face crop (Target rich environment for Face2Face artifacts).

### 3. Model Architecture
Both models utilize **Transfer Learning** via `EfficientNetB1` (ImageNet weights) wrapped in a `TimeDistributed` layer, followed by a `Bidirectional LSTM`.

* **Baseline (Control):** Uses `GlobalAveragePooling1D` to average features across all frames.
* **Proposed (Attention):** Uses a learned **Temporal Attention Mechanism** (`Dense` -> `Softmax` -> `Multiply`) to assign importance weights to specific frames, allowing the model to focus on fleeting "micro-jitters" while ignoring stable frames.

## 🚀 Usage

### 0. Create Environment
Create an environment and install requirements as needed.
```bash
pip install -r requirements.txt
```
Set environment to block conflicting local libraries (if on cluster)
```bash
export PYTHONNOUSERSITE=1
```

### 1. Pre-process the Data
First, convert the raw videos into NumPy arrays. This step requires the `haarcascade_frontalface_default.xml` file.
```bash
python preprocess.py
```

Output: Creates Processed_Data_Full_HighRes, Processed_Data_Eyes_HighRes, and Processed_Data_Mouth_HighRes.

### 2. Run the Experiment
Configure the experiment in run.py by changing the REGION_TO_CROP variable ('full', 'eyes', or 'mouth') and MODEL_NAME. Then, start training requires GPU based on file config).

```bash
python run.py
```
