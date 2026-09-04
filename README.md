<h1 align="center">🎙️ SpeechSense AI</h1>

<p align="center">
  <b>EMOTION</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Streamlit-Framework-red?style=for-the-badge&logo=streamlit&logoColor=white">
  <img src="https://img.shields.io/badge/Librosa-Audio%20Processing-purple?style=for-the-badge">
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange?style=for-the-badge&logo=scikit-learn&logoColor=white">
  <img src="https://img.shields.io/badge/Random%20Forest-Classifier-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/RAVDESS-Dataset-yellow?style=for-the-badge">
</p>

<p align="center">
  <b>🎭 AI-powered Speech Emotion Recognition system that analyzes speech audio and predicts emotional patterns using Machine Learning.</b>
</p>

<p align="center">
  Built with Python, Librosa, Scikit-learn and Streamlit.
</p>

<p align="center">
  🚀 <a href="https://emotion-recognition-from-speech-4cigdockiyfpmnbysb4z5y.streamlit.app/"><b>Live Demo</b></a>
</p>

---

## 📌 About The Project

**SpeechSense AI** is an intelligent Speech Emotion Recognition application developed using **Python, Machine Learning, Librosa and Streamlit**.

The system analyzes speech audio and predicts the most likely emotional pattern from the speaker's voice.

The project uses the **RAVDESS Speech Emotion Dataset** and extracts important audio features including **MFCC, Chroma and Mel Spectrogram features**.

A **Random Forest Classifier** is trained on these features to recognize seven different emotions.

The application provides an interactive Streamlit dashboard with audio analysis, emotion prediction, confidence scores, visualizations, batch analysis, model performance and prediction history.

> ⚠️ **Important:** The model predicts acoustic patterns in speech. It does not directly determine a person's actual internal emotional state.

---

## 🎭 Supported Emotions

| Emotion     | Description                       |
| ----------- | --------------------------------- |
| 😠 Angry    | Anger-related speech patterns     |
| 🤢 Disgust  | Disgust-related speech patterns   |
| 😨 Fear     | Fear-related speech patterns      |
| 😊 Happy    | Happiness-related speech patterns |
| 😐 Neutral  | Neutral speech patterns           |
| 😢 Sad      | Sadness-related speech patterns   |
| 😲 Surprise | Surprise-related speech patterns  |

---

## ✨ Key Features

### 🎧 Audio Input

Users can provide speech through:

* 📁 Audio file upload
* 🎙️ Microphone recording
* 🔊 Demo audio

**Supported Formats:**

* WAV
* MP3
* M4A

### 🧠 Emotion Prediction

The trained **Random Forest Classifier** predicts one of seven supported emotions.

The application displays:

* 🎯 Predicted emotion
* 📊 Confidence score
* 🥇 Top prediction
* 🥈 Second-best prediction
* 📈 Top 3 predictions
* 🔍 Prediction reliability

### 📊 Audio Analysis

Speech audio is analyzed using:

* MFCC
* Chroma
* Mel Spectrogram
* RMS Energy
* Zero Crossing Rate
* Spectral features
* Audio duration
* Sample rate
* Peak amplitude

### 🎚️ Audio Health Analysis

SpeechSense AI checks:

* ⏱️ Audio duration
* 🎵 Sample rate
* 🔊 RMS energy
* 📈 Peak amplitude
* ⚠️ Clipping detection
* 🔇 Silence ratio
* 📶 Noise level
* ⭐ Audio quality score

### 📈 Audio Visualizations

The dashboard provides:

* 🌊 Waveform
* 🎵 Mel Spectrogram
* 📊 MFCC visualization
* 📡 Acoustic statistics
* 📈 Feature analysis

### 🤖 AI Insights

The application provides:

* Emotion interpretation
* Emotion intensity
* Acoustic energy
* Prediction strength
* Audio health
* Noise detection
* Smart recommendations

### 🔬 Explainable AI

SpeechSense AI provides:

* Feature importance
* Emotion probabilities
* Prediction confidence
* Top predictions
* Emotion radar chart
* Feature explorer

### 📦 Batch Emotion Analysis

Users can analyze multiple audio files together.

Batch results include:

* File name
* Predicted emotion
* Confidence
* Audio statistics

Results can be exported as CSV.

### 📊 Model Performance

The application provides:

* Accuracy
* Classification report
* Confusion matrix
* Emotion-wise performance
* Model performance visualization

### 🕒 Prediction History

Users can review:

* Previous predictions
* Confidence scores
* Prediction timestamps
* Audio analysis results

### 📄 Export Reports

Available exports include:

* 📊 Prediction CSV
* 📦 Batch Analysis CSV
* 🕒 History CSV
* 📄 PDF Report

---

## 🏆 Model Performance

The SpeechSense AI model achieved:

```text
🎯 Test Accuracy: 94.10%
```

### Dataset Split

```text
Training Samples: 2,304
Testing Samples:    576
Total Samples:     2,880
```

The reported accuracy represents performance on the project's held-out test data.

---

## 🧠 Machine Learning Model

### 🌲 Random Forest Classifier

SpeechSense AI uses a **Random Forest Classifier** for emotion classification.

Random Forest combines multiple decision trees to make a final prediction.

It works with the extracted acoustic features and also provides useful feature-importance information for explainability.

---

## 🎵 Feature Extraction

The audio processing pipeline extracts **92 features** from each audio sample.

| Feature   | Number |
| --------- | -----: |
| 🎵 MFCC   |     40 |
| 🎼 Chroma |     12 |
| 🌈 Mel    |     40 |
| **Total** | **92** |

### 🔄 Feature Pipeline

```text
🎧 Speech Audio
       ↓
🔊 Audio Loading
       ↓
🎵 Librosa Processing
       ↓
┌──────────────────────┐
│ MFCC       → 40      │
│ Chroma     → 12      │
│ Mel        → 40      │
└──────────────────────┘
       ↓
📊 92 Audio Features
       ↓
⚙️ Feature Scaling
       ↓
🌲 Random Forest
       ↓
🎭 Emotion Prediction
       ↓
📈 Confidence & Insights
```

---

## 🔄 How SpeechSense AI Works

```text
              👤 User
                 │
          🎧 Audio Input
                 │
        ┌────────┴────────┐
        │                 │
     📁 Upload         🎙️ Record
        │                 │
        └────────┬────────┘
                 ↓
          🔊 Audio Processing
                 ↓
        🎵 Feature Extraction
                 ↓
       MFCC + Chroma + Mel
                 ↓
          📊 Feature Vector
                 ↓
          ⚙️ Feature Scaling
                 ↓
        🌲 Random Forest Model
                 ↓
          🎯 Emotion Prediction
                 ↓
       ┌─────────┴─────────┐
       │                   │
   📊 Confidence       🤖 AI Insights
       │                   │
       └─────────┬─────────┘
                 ↓
          📈 Visualization
                 ↓
          👤 User Results
```

---

## 🛠️ Technologies Used

| Technology       | Purpose                               |
| ---------------- | ------------------------------------- |
| 🐍 Python        | Application Development               |
| 🎈 Streamlit     | Web Application Interface             |
| 🎵 Librosa       | Audio Processing & Feature Extraction |
| 🧠 Scikit-learn  | Machine Learning                      |
| 🌲 Random Forest | Emotion Classification                |
| 🔢 NumPy         | Numerical Processing                  |
| 📊 Pandas        | Data Processing                       |
| 📈 Matplotlib    | Data Visualization                    |
| 💾 Joblib        | Model Serialization                   |
| 🎧 RAVDESS       | Speech Emotion Dataset                |
| 🐙 GitHub        | Version Control                       |

---

## 📂 Project Structure

```text
Emotion-Recognition-from-Speech/
│
├── app.py
├── feature_extraction.py
├── organize_dataset.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── emotion_model.pkl
│   ├── label_encoder.pkl
│   └── scaler.pkl
│
└── dataset/
    └── RAVDESS Audio Dataset
```

> The dataset and audio files are excluded from the GitHub repository using `.gitignore`.

---

## 📄 Main Files

### `app.py`

Main Streamlit application containing:

* Dashboard
* Audio input
* Emotion prediction
* Audio analysis
* AI insights
* Batch analysis
* Model performance
* Prediction history
* Report generation

### `feature_extraction.py`

Handles audio feature extraction using Librosa.

Extracts:

* MFCC
* Chroma
* Mel features
* Acoustic features

### `train_model.py`

Responsible for:

* Loading training data
* Extracting features
* Scaling features
* Training Random Forest
* Evaluating the model
* Saving trained model files

### `organize_dataset.py`

Organizes the RAVDESS dataset into emotion-based folders for easier training and processing.

### `models/`

Contains:

```text
emotion_model.pkl
label_encoder.pkl
scaler.pkl
```

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Tehmina124/Emotion-Recognition-from-Speech.git
```

### 2️⃣ Open the Project Folder

```bash
cd Emotion-Recognition-from-Speech
```

### 3️⃣ Create Virtual Environment

```bash
python -m venv .venv
```

### 4️⃣ Activate Virtual Environment

For Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 5️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 6️⃣ Run SpeechSense AI

```bash
python -m streamlit run app.py
```

### 7️⃣ Open in Browser

```text
http://localhost:8501
```

---

## 🌐 Live Demo

<p align="center">
  🚀 <a href="https://emotion-recognition-from-speech-4cigdockiyfpmnbysb4z5y.streamlit.app/">
    <b>Launch SpeechSense AI</b>
  </a>
</p>

SpeechSense AI is deployed using **Streamlit** and can be accessed online.

---

## 🎯 Project Objectives

The main objectives of this project were:

* Build a practical Speech Emotion Recognition system.
* Apply Machine Learning to speech/audio data.
* Understand audio feature extraction.
* Work with the RAVDESS dataset.
* Extract MFCC, Chroma and Mel features.
* Train a Random Forest classifier.
* Build an interactive Streamlit dashboard.
* Implement explainable prediction features.
* Analyze audio quality and acoustic properties.
* Support batch audio analysis.
* Generate downloadable reports.
* Practice GitHub project management.

---

## 💡 What I Learned

Through this project, I gained practical experience in:

* Python development
* Machine Learning
* Speech Emotion Recognition
* Audio processing
* Librosa
* MFCC feature extraction
* Chroma features
* Mel Spectrograms
* Random Forest classification
* Feature scaling
* Model evaluation
* Confusion matrices
* Explainable AI
* Streamlit application development
* Data visualization
* Batch processing
* GitHub repository management

This project helped me understand how **audio signals can be transformed into machine-learning features and used for emotion classification**.

---

## 🔮 Future Improvements

Future versions of SpeechSense AI can include:

* 🤖 Deep Learning models
* 🧠 CNN-based audio classification
* 🔥 LSTM / BiLSTM models
* 🎙️ Real-time emotion detection
* 🌍 Multi-language emotion recognition
* 👥 Speaker-independent evaluation
* 📱 Mobile application
* ☁️ Advanced cloud deployment
* 🎧 Better noise reduction
* 📊 Advanced analytics
* 🧠 Transformer-based speech models
* 🔐 User authentication
* 💾 Persistent prediction database

---

## 🧪 Example Prediction

```text
🎯 Predicted Emotion: Neutral

📊 Confidence: 74.67%

🥇 Neutral: 74.67%
🥈 Sad: 6.00%
🥉 Fear: 5.00%

🔍 Top-2 Separation: 68.67%

⭐ Reliability: High

🎚️ Emotion Intensity: 66.7 / 100

🔊 Acoustic Energy: 24.1 / 100

📈 Prediction Strength: 74.7 / 100
```

---

## 🛡️ Responsible AI

Speech emotion recognition should be used carefully.

Prediction quality can be affected by:

* 🎙️ Microphone quality
* 🔊 Background noise
* 👤 Speaker characteristics
* 🗣️ Accent
* 🌍 Language
* ⏱️ Audio duration
* 🏠 Recording environment
* 🔊 Volume and clipping

The reported **94.10% accuracy** represents evaluation on the project's test data and does not guarantee the same performance in real-world situations.

### ⚠️ Intended Use

SpeechSense AI is intended for:

* 🎓 Educational demonstrations
* 🔬 Machine Learning research
* 🎧 Speech analysis experiments
* 💻 AI/ML portfolio projects

It should **not** be used for:

* Medical or mental-health diagnosis
* Employment decisions
* Legal decisions
* High-stakes profiling
* Determining a person's actual psychological state

---

## 👩‍💻 About Me

### Tehmina Anwar

<p align="center">
  <b>🤖 AI Developer • AI/ML Engineer • Python Developer • Generative AI</b>
</p>

I am a Bachelor of Science in Artificial Intelligence student passionate about building practical and intelligent AI applications.

My interests include:

* 🐍 Python
* 🧠 Machine Learning
* 🤖 Generative AI
* 🧠 Large Language Models
* 📚 RAG
* 📝 Natural Language Processing
* 👁️ Computer Vision
* 💻 AI Application Development

I enjoy turning AI and Machine Learning concepts into practical projects and user-friendly applications.

---

## 🔗 Connect With Me

### 💻 GitHub

https://github.com/Tehmina124

### 🔗 LinkedIn

https://www.linkedin.com/in/tehmina-anwar-77b8a8414/

### 🌐 Portfolio

https://tehmina-portfolio-five.vercel.app/

---

## ⭐ Support

If you found this project useful or interesting, please consider giving the repository a ⭐ **Star** on GitHub.

---

<p align="center">
  <b>Built with ❤️ using Python, Machine Learning, Librosa and Streamlit</b>
</p>

<p align="center">
  © 2026 Tehmina Anwar | SpeechSense AI</p>
