
import os
import numpy as np
import librosa
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 🎙️ Emotion Recognition from Speech
# 👩‍💻 Created by: Tehmina Anwar
# ============================================================

DATASET_PATH = "dataset"
MODEL_DIR = "models"

# ============================================================
# 🎭 RAVDESS EMOTION MAPPING
# ============================================================

EMOTION_MAP = {
    "01": "neutral",
    "02": "neutral",      # Calm → Neutral
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise"
}

# ============================================================
# 🔬 FEATURE EXTRACTION
# ============================================================
# IMPORTANT:
# This creates EXACTLY 92 features:
#
# MFCC   = 40
# Chroma = 12
# Mel    = 40
# ----------------
# TOTAL  = 92
# ============================================================

def extract_features(file_path):

    try:

        # ----------------------------------------------------
        # Load audio
        # Same preprocessing used during training
        # ----------------------------------------------------

        audio, sample_rate = librosa.load(
            file_path,
            sr=22050,
            duration=3,
            offset=0.5
        )

        # ----------------------------------------------------
        # MFCC
        # 40 features
        # ----------------------------------------------------

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        )

        mfcc_mean = np.mean(
            mfcc,
            axis=1
        )

        # ----------------------------------------------------
        # Chroma
        # 12 features
        # ----------------------------------------------------

        stft = librosa.stft(audio)

        chroma = librosa.feature.chroma_stft(
            S=np.abs(stft),
            sr=sample_rate
        )

        chroma_mean = np.mean(
            chroma,
            axis=1
        )

        # ----------------------------------------------------
        # Mel Spectrogram
        # 40 features
        # ----------------------------------------------------

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_mels=40
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        mel_mean = np.mean(
            mel_db,
            axis=1
        )

        # ----------------------------------------------------
        # Combine all features
        # 40 + 12 + 40 = 92
        # ----------------------------------------------------

        features = np.hstack([
            mfcc_mean,
            chroma_mean,
            mel_mean
        ])

        # Safety check
        if len(features) != 92:

            print(
                f"⚠️ Unexpected feature count: {len(features)}"
            )

            return None

        return features

    except Exception as e:

        print(
            f"❌ Error processing: {file_path}"
        )

        print(
            f"   Error: {e}"
        )

        return None


# ============================================================
# 🚀 START TRAINING
# ============================================================

print("\n" + "=" * 65)
print("🎙️ EMOTION RECOGNITION FROM SPEECH")
print("=" * 65)

print("\n📂 Searching dataset...")
print(f"Dataset path: {DATASET_PATH}")


# ============================================================
# 📦 DATA STORAGE
# ============================================================

X = []
y = []

total_files = 0
processed_files = 0
failed_files = 0

emotion_counts = {
    emotion: 0
    for emotion in sorted(set(EMOTION_MAP.values()))
}


# ============================================================
# 🔎 SEARCH DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):

    print(
        f"\n❌ Dataset folder not found: {DATASET_PATH}"
    )

    print(
        "\nPlease make sure your project has:"
    )

    print(
        "Emotion-Recognition-from-Speech/"
    )

    print(
        "└── dataset/"
    )

    raise SystemExit


for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        # Only WAV files
        if not file.lower().endswith(".wav"):
            continue

        total_files += 1

        file_path = os.path.join(
            root,
            file
        )

        # ----------------------------------------------------
        # Read RAVDESS filename
        #
        # Example:
        # 03-01-05-01-02-01-12.wav
        #
        # parts[2] = emotion code
        # ----------------------------------------------------

        parts = file.split("-")

        if len(parts) < 3:

            print(
                f"⚠️ Invalid filename: {file}"
            )

            failed_files += 1

            continue

        emotion_code = parts[2]

        emotion = EMOTION_MAP.get(
            emotion_code
        )

        if emotion is None:

            print(
                f"⚠️ Unknown emotion code: {emotion_code}"
            )

            failed_files += 1

            continue

        # ----------------------------------------------------
        # Extract features
        # ----------------------------------------------------

        features = extract_features(
            file_path
        )

        if features is None:

            failed_files += 1

            continue

        # ----------------------------------------------------
        # Store features and label
        # ----------------------------------------------------

        X.append(features)
        y.append(emotion)

        emotion_counts[emotion] += 1
        processed_files += 1

        # Progress
        if processed_files % 100 == 0:

            print(
                f"✅ Processed: {processed_files} files"
            )


# ============================================================
# 📊 DATASET SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("📊 DATASET SUMMARY")
print("=" * 65)

print(
    f"\n🎵 Total WAV files found: {total_files}"
)

print(
    f"✅ Successfully processed: {processed_files}"
)

print(
    f"❌ Failed/skipped files: {failed_files}"
)

print("\n🎭 Emotion Distribution:")

for emotion, count in sorted(
    emotion_counts.items()
):

    print(
        f"   {emotion:<10}: {count}"
    )


# ============================================================
# ❌ CHECK DATA
# ============================================================

if len(X) == 0:

    print(
        "\n❌ ERROR: No audio features were extracted."
    )

    print(
        "\nPlease check your dataset folder."
    )

    raise SystemExit


# ============================================================
# 🔢 CONVERT TO NUMPY
# ============================================================

X = np.array(X)
y = np.array(y)

print(
    f"\n🔢 Feature shape: {X.shape}"
)

print(
    f"🏷️ Total labels: {len(y)}"
)

# Confirm 92 features
if X.shape[1] != 92:

    print(
        f"\n❌ ERROR: Expected 92 features, "
        f"but got {X.shape[1]}"
    )

    raise SystemExit


print(
    "\n✅ Feature count confirmed: 92"
)

print(
    "   MFCC:   40"
)

print(
    "   Chroma: 12"
)

print(
    "   Mel:    40"
)

print(
    "   Total:  92"
)


# ============================================================
# 🏷️ LABEL ENCODING
# ============================================================

print("\n🏷️ Encoding emotion labels...")

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(
    y
)

print(
    "\n🎭 Emotion Labels:"
)

for number, emotion in enumerate(
    label_encoder.classes_
):

    print(
        f"   {number} → {emotion}"
    )


# ============================================================
# 📚 TRAIN / TEST SPLIT
# ============================================================

print(
    "\n📚 Splitting dataset..."
)

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y_encoded,

    test_size=0.20,

    random_state=42,

    stratify=y_encoded
)

print(
    f"\n📚 Training samples: {len(X_train)}"
)

print(
    f"🧪 Testing samples:  {len(X_test)}"
)


# ============================================================
# ⚙️ FEATURE SCALING
# ============================================================

print(
    "\n⚙️ Scaling features..."
)

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)

print(
    "✅ Feature scaling completed!"
)


# ============================================================
# 🤖 RANDOM FOREST CLASSIFIER
# ============================================================

print(
    "\n🤖 Training Random Forest..."
)

model = RandomForestClassifier(

    n_estimators=300,

    random_state=42,

    n_jobs=-1,

    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)

print(
    "✅ Random Forest training completed!"
)


# ============================================================
# 🔮 PREDICTIONS
# ============================================================

print(
    "\n🔮 Making predictions..."
)

y_pred = model.predict(
    X_test
)


# ============================================================
# 🎯 MODEL ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

accuracy_percent = accuracy * 100


# ============================================================
# 📊 MODEL RESULTS
# ============================================================

print("\n" + "=" * 65)
print("📊 MODEL RESULTS")
print("=" * 65)

print(
    f"\n🎯 Test Accuracy: {accuracy_percent:.2f}%"
)

print(
    f"📚 Training Samples: {len(X_train)}"
)

print(
    f"🧪 Testing Samples: {len(X_test)}"
)

print(
    f"🔢 Audio Features: {X.shape[1]}"
)


# ============================================================
# 📋 CLASSIFICATION REPORT
# ============================================================

print(
    "\n📋 Classification Report:\n"
)

report = classification_report(

    y_test,

    y_pred,

    target_names=label_encoder.classes_

)

print(
    report
)


# ============================================================
# 🔲 CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print(
    "\n🔲 Confusion Matrix:"
)

print(
    cm
)


# ============================================================
# 💾 SAVE MODEL FILES
# ============================================================

print(
    "\n💾 Saving model files..."
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

# Save Random Forest
joblib.dump(
    model,
    os.path.join(
        MODEL_DIR,
        "emotion_model.pkl"
    )
)

# Save scaler
joblib.dump(
    scaler,
    os.path.join(
        MODEL_DIR,
        "scaler.pkl"
    )
)

# Save label encoder
joblib.dump(
    label_encoder,
    os.path.join(
        MODEL_DIR,
        "label_encoder.pkl"
    )
)


# ============================================================
# ✅ VERIFY SAVED FILES
# ============================================================

print("\n" + "=" * 65)
print("💾 MODEL FILES SAVED")
print("=" * 65)

print(
    "\n✅ models/emotion_model.pkl"
)

print(
    "✅ models/scaler.pkl"
)

print(
    "✅ models/label_encoder.pkl"
)


# ============================================================
# 🎉 FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 65)

print(
    f"\n🎯 Final Accuracy: {accuracy_percent:.2f}%"
)

print(
    f"🔢 Features: {X.shape[1]}"
)

print(
    f"📚 Training Samples: {len(X_train)}"
)

print(
    f"🧪 Testing Samples: {len(X_test)}"
)

print(
    "\n🎭 Classes:"
)

for emotion in label_encoder.classes_:

    print(
        f"   • {emotion}"
    )

print(
    "\n🚀 Your SpeechSense AI model is ready!"
)

print(
    "=" * 65
)