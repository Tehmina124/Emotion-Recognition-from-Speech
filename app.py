import os
import hashlib
from io import BytesIO
from datetime import datetime

import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 🎙️ SPEECHSENSE AI
# AI-Powered Speech Emotion Recognition
# Created by: Tehmina Anwar
# ============================================================


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SpeechSense AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

TEST_ACCURACY = 94.10
TRAINING_SAMPLES = 2304
TESTING_SAMPLES = 576

TOTAL_FEATURES = 92
MFCC_FEATURES = 40
CHROMA_FEATURES = 12
MEL_FEATURES = 40

MODEL_NAME = "Random Forest Classifier"
DATASET_NAME = "RAVDESS Speech Emotion Dataset"

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

EMOTION_EMOJIS = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😊",
    "neutral": "😐",
    "sad": "😢",
    "surprise": "😲",
}

EMOTION_DESCRIPTIONS = {
    "angry": (
        "The audio contains acoustic patterns commonly associated "
        "with anger, such as stronger energy or intensity."
    ),
    "disgust": (
        "The audio contains acoustic patterns that may be associated "
        "with disgust."
    ),
    "fear": (
        "The audio contains acoustic patterns that may be associated "
        "with fear or vocal tension."
    ),
    "happy": (
        "The audio contains acoustic patterns associated with "
        "positive or energetic expression."
    ),
    "neutral": (
        "The audio appears relatively balanced without a strong "
        "dominant emotional pattern."
    ),
    "sad": (
        "The audio contains acoustic patterns that may be associated "
        "with sadness or lower emotional energy."
    ),
    "surprise": (
        "The audio contains acoustic patterns that may be associated "
        "with sudden or heightened expression."
    ),
}


# ============================================================
# MODEL PERFORMANCE
# ============================================================

CLASSIFICATION_REPORT = pd.DataFrame(
    {
        "Emotion": [
            "Angry",
            "Disgust",
            "Fear",
            "Happy",
            "Neutral",
            "Sad",
            "Surprise",
        ],
        "Precision": [
            0.92,
            0.97,
            0.93,
            0.97,
            0.91,
            1.00,
            0.92,
        ],
        "Recall": [
            0.95,
            0.95,
            0.97,
            0.92,
            1.00,
            0.82,
            0.95,
        ],
        "F1-Score": [
            0.94,
            0.96,
            0.95,
            0.95,
            0.95,
            0.90,
            0.94,
        ],
    }
)


CONFUSION_MATRIX = np.array(
    [
        [72, 2, 0, 0, 0, 0, 2],
        [2, 73, 0, 0, 0, 0, 2],
        [0, 0, 75, 0, 0, 0, 2],
        [2, 0, 2, 71, 2, 0, 0],
        [0, 0, 0, 0, 115, 0, 0],
        [2, 0, 2, 0, 10, 63, 0],
        [0, 0, 2, 2, 0, 0, 73],
    ]
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "emotion_model.pkl",
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl",
)

ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "label_encoder.pkl",
)

TEST_AUDIO_FOLDER = os.path.join(
    BASE_DIR,
    "test_audio",
)

TEMP_FOLDER = os.path.join(
    BASE_DIR,
    "temp_audio",
)

os.makedirs(TEMP_FOLDER, exist_ok=True)


# ============================================================
# CHART RENDERER
# ============================================================

def show_figure(fig, width="stretch"):
    """Render Matplotlib figures as PNGs to avoid Streamlit canvas/SVG artifacts."""
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    st.image(buffer, width=width)
    plt.close(fig)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.main {
    background: linear-gradient(
        135deg,
        #f8fafc 0%,
        #eef2ff 50%,
        #f8fafc 100%
    );
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #111827 0%,
        #1e1b4b 100%
    );
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.hero {
    padding: 2rem;
    border-radius: 25px;
    background: linear-gradient(
        135deg,
        #312e81,
        #4f46e5,
        #7c3aed
    );
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 12px 30px rgba(79,70,229,0.25);
}

.hero h1 {
    font-size: 3rem;
    margin-bottom: 0.4rem;
}

.hero p {
    font-size: 1.1rem;
    opacity: 0.92;
}

.metric-card {
    padding: 1.2rem;
    border-radius: 18px;
    background: white;
    box-shadow: 0 6px 20px rgba(15,23,42,0.08);
    border: 1px solid #e5e7eb;
}

.big-result {
    padding: 2rem;
    border-radius: 25px;
    text-align: center;
    background: linear-gradient(
        135deg,
        #eef2ff,
        #faf5ff
    );
    border: 2px solid #c7d2fe;
}

.big-emotion {
    font-size: 4rem;
    font-weight: 800;
}

.confidence-number {
    font-size: 2.7rem;
    font-weight: 800;
}

.info-card {
    padding: 1.3rem;
    border-radius: 18px;
    background: white;
    border: 1px solid #e5e7eb;
    box-shadow: 0 5px 18px rgba(15,23,42,0.06);
}

.footer {
    text-align: center;
    padding: 2rem 0;
    color: #64748b;
}

.sidebar-nav {
    display: flex !important;
    flex-direction: column !important;
    gap: 0.55rem;
    margin: 0.55rem 0 1rem 0;
}

.sidebar-nav a {
    display: flex;
    align-items: center;
    width: 100%;
    box-sizing: border-box;
    padding: 0.72rem 0.85rem;
    margin: 0;
    border-radius: 12px;
    text-decoration: none !important;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.10);
    color: white !important;
    font-weight: 650;
    line-height: 1.25;
    transition: all 0.2s ease;
}

.sidebar-nav a:hover {
    background: rgba(255,255,255,0.16);
    border-color: rgba(255,255,255,0.24);
    transform: translateX(2px);
}

/* Remove Streamlit heading/action icons that can show up as [svg]. */
h1 a, h2 a, h3 a, h4 a, h5 a, h6 a,
[data-testid="stHeaderActionElements"],
[data-testid="stHeaderActionElements"] *,
[data-testid="stHeadingWithActionElements"] a,
[data-testid="stHeadingWithActionElements"] button,
[data-testid="stHeadingWithActionElements"] svg,
[data-testid="stHeadingWithActionElements"] [aria-hidden="true"],
[data-testid="stMarkdownContainer"] h1 a,
[data-testid="stMarkdownContainer"] h2 a,
[data-testid="stMarkdownContainer"] h3 a,
[data-testid="stMarkdownContainer"] h4 a,
[data-testid="stMarkdownContainer"] h5 a,
[data-testid="stMarkdownContainer"] h6 a {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    pointer-events: none !important;
}

/* Streamlit can inject SVG/canvas accessibility elements around charts.
   SpeechSense uses static Matplotlib figures, so these helper layers are not needed. */
div.stButton > button {
    border-radius: 12px;
    font-weight: 700;
    min-height: 2.7rem;
}

[data-testid="stMetric"] {
    background: white;
    padding: 1rem;
    border-radius: 15px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 15px rgba(15,23,42,0.05);
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "history": [],
    "selected_audio_path": None,
    "selected_audio_filename": None,
    "selected_audio_id": None,
    "prediction_result": None,
    "batch_results": None,
    "timeline_results": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_models():
    required_files = [
        (MODEL_PATH, "emotion_model.pkl"),
        (SCALER_PATH, "scaler.pkl"),
        (ENCODER_PATH, "label_encoder.pkl"),
    ]

    missing = []

    for path, name in required_files:
        if not os.path.exists(path):
            missing.append(
                f"{name}\nExpected: {path}"
            )

    if missing:
        raise FileNotFoundError(
            "Missing model files:\n\n"
            + "\n\n".join(missing)
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)

    return model, scaler, label_encoder


try:
    model, scaler, label_encoder = load_models()
    MODEL_READY = True

except Exception as error:
    MODEL_READY = False

    st.error(
        "❌ SpeechSense AI model files could not be loaded."
    )

    st.code(
        f"""
Required folder structure:

SpeechSense-AI/
│
├── app.py
│
├── models/
│   ├── emotion_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
└── test_audio/

Error:
{error}
"""
    )

    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_filename(filename):
    """
    Creates a safe temporary filename.
    """
    base = os.path.basename(filename)

    clean = "".join(
        char if char.isalnum() or char in "._-"
        else "_"
        for char in base
    )

    return clean


def file_hash(uploaded_file):
    """
    Generates a unique ID for uploaded audio.
    """
    return hashlib.md5(
        uploaded_file.getvalue()
    ).hexdigest()


# ============================================================
# FEATURE EXTRACTION
# IMPORTANT:
# MUST MATCH TRAINING PIPELINE
# ============================================================

def extract_features(file_path, offset=0.5):

    audio, sample_rate = librosa.load(
        file_path,
        sr=22050,
        duration=3,
        offset=offset,
    )

    if audio is None or len(audio) == 0:
        raise ValueError(
            "The selected audio does not contain readable audio."
        )

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40,
    )

    mfcc_mean = np.mean(
        mfcc,
        axis=1,
    )

    # --------------------------------------------------------
    # Chroma
    # --------------------------------------------------------

    stft = librosa.stft(
        audio
    )

    chroma = librosa.feature.chroma_stft(
        S=np.abs(stft),
        sr=sample_rate,
    )

    chroma_mean = np.mean(
        chroma,
        axis=1,
    )

    # --------------------------------------------------------
    # Mel Spectrogram
    # --------------------------------------------------------

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=40,
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    mel_mean = np.mean(
        mel_db,
        axis=1,
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    features = np.hstack(
        [
            mfcc_mean,
            chroma_mean,
            mel_mean,
        ]
    )

    if len(features) != TOTAL_FEATURES:
        raise ValueError(
            f"Feature extraction returned "
            f"{len(features)} features instead of "
            f"{TOTAL_FEATURES}."
        )

    return features


# ============================================================
# AUDIO INFORMATION
# ============================================================

def get_audio_information(file_path):

    audio, sample_rate = librosa.load(
        file_path,
        sr=None,
        mono=True,
    )

    if audio is None or len(audio) == 0:
        raise ValueError(
            "No readable audio data was found."
        )

    duration = librosa.get_duration(
        y=audio,
        sr=sample_rate,
    )

    return audio, sample_rate, duration


# ============================================================
# AUDIO QUALITY
# ============================================================

def calculate_audio_quality(
    audio,
    sample_rate,
    duration,
):

    rms = float(
        np.sqrt(
            np.mean(
                np.square(audio)
            )
        )
    )

    peak = float(
        np.max(
            np.abs(audio)
        )
    )

    clipping_ratio = float(
        np.mean(
            np.abs(audio) >= 0.99
        )
    )

    silence_ratio = float(
        np.mean(
            np.abs(audio) < 0.005
        )
    )

    score = 100.0

    if duration < 1:
        score -= 45
    elif duration < 2:
        score -= 25
    elif duration < 3:
        score -= 10

    if rms < 0.01:
        score -= 25
    elif rms < 0.02:
        score -= 10

    if clipping_ratio > 0.01:
        score -= 20
    elif clipping_ratio > 0.001:
        score -= 8

    if silence_ratio > 0.80:
        score -= 20
    elif silence_ratio > 0.60:
        score -= 8

    if sample_rate < 16000:
        score -= 8

    score = float(
        np.clip(
            score,
            0,
            100,
        )
    )

    if score >= 85:
        label = "Excellent"
    elif score >= 70:
        label = "Good"
    elif score >= 50:
        label = "Fair"
    else:
        label = "Poor"

    return {
        "score": score,
        "label": label,
        "rms": rms,
        "peak": peak,
        "clipping_ratio": clipping_ratio,
        "silence_ratio": silence_ratio,
    }


# ============================================================
# RELIABILITY
# ============================================================

def confidence_level(
    confidence,
    margin,
):

    if confidence >= 70 and margin >= 35:
        return (
            "High Reliability",
            "🟢",
            "The top prediction is strongly separated "
            "from the runner-up.",
        )

    if confidence >= 50 and margin >= 15:
        return (
            "Moderate Reliability",
            "🟡",
            "The prediction is reasonably separated, "
            "but should still be interpreted cautiously.",
        )

    return (
        "Low Reliability",
        "🔴",
        "The prediction is uncertain because competing "
        "emotion probabilities are relatively close.",
    )


# ============================================================
# SMART RECOMMENDATION
# ============================================================

def smart_recommendation(
    confidence,
    margin,
    quality_score,
):

    if (
        confidence >= 70
        and margin >= 35
        and quality_score >= 70
    ):
        return (
            "Strong Result",
            "The prediction is supported by high confidence, "
            "good separation from the second prediction, "
            "and acceptable recording quality.",
        )

    if (
        confidence >= 70
        and margin >= 35
        and quality_score < 70
    ):
        return (
            "Strong Prediction / Improve Audio",
            "The model prediction is strong, but improving "
            "recording quality may make the result more reliable.",
        )

    if (
        confidence >= 50
        and margin >= 15
        and quality_score >= 50
    ):
        return (
            "Moderate Result",
            "The model has a reasonable preference, but "
            "the result should be interpreted cautiously.",
        )

    return (
        "Needs Review",
        "The prediction is uncertain. Consider a cleaner, "
        "longer recording and review the probability distribution.",
    )


# ============================================================
# LABEL CONVERSION
# ============================================================

def decode_prediction(prediction_encoded):

    """
    Handles both numeric and string class labels.
    """

    try:
        decoded = label_encoder.inverse_transform(
            [prediction_encoded]
        )[0]

        return str(decoded)

    except Exception:

        return str(prediction_encoded)


def decode_classes(class_ids):

    """
    Safely converts model classes into emotion names.
    """

    try:

        decoded = label_encoder.inverse_transform(
            class_ids
        )

        return [
            str(label)
            for label in decoded
        ]

    except Exception:

        return [
            str(label)
            for label in class_ids
        ]


# ============================================================
# PREDICTION
# ============================================================

def predict_audio(
    file_path,
    offset=0.5,
):

    features = extract_features(
        file_path,
        offset=offset,
    )

    feature_array = features.reshape(
        1,
        -1,
    )

    expected_features = getattr(
        scaler,
        "n_features_in_",
        TOTAL_FEATURES,
    )

    if feature_array.shape[1] != expected_features:
        raise ValueError(
            f"Model expects {expected_features} features, "
            f"but received {feature_array.shape[1]}."
        )

    scaled_features = scaler.transform(
        feature_array
    )

    prediction_encoded = model.predict(
        scaled_features
    )[0]

    prediction = decode_prediction(
        prediction_encoded
    )

    if not hasattr(
        model,
        "predict_proba",
    ):
        raise ValueError(
            "The loaded model does not support probability prediction."
        )

    probabilities_array = model.predict_proba(
        scaled_features
    )[0]

    class_ids = model.classes_

    emotion_labels = decode_classes(
        class_ids
    )

    probability_dict = {}

    for label, probability in zip(
        emotion_labels,
        probabilities_array,
    ):

        probability_dict[
            str(label).lower()
        ] = float(
            probability * 100
        )

    # Make sure all expected emotions exist
    for emotion in EMOTIONS:

        if emotion not in probability_dict:
            probability_dict[emotion] = 0.0

    ordered = sorted(
        probability_dict.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    prediction = prediction.lower()

    if prediction not in probability_dict:

        prediction = ordered[0][0]

    confidence = probability_dict[
        prediction
    ]

    if len(ordered) > 1:

        margin = (
            ordered[0][1]
            - ordered[1][1]
        )

    else:

        margin = confidence

    (
        reliability,
        reliability_icon,
        reliability_text,
    ) = confidence_level(
        confidence,
        margin,
    )

    return {
        "emotion": prediction,
        "confidence": confidence,
        "probabilities": probability_dict,
        "margin": margin,
        "reliability": reliability,
        "reliability_icon": reliability_icon,
        "reliability_text": reliability_text,
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance():

    if not hasattr(
        model,
        "feature_importances_",
    ):
        return None, None

    importances = np.asarray(
        model.feature_importances_
    )

    if len(importances) != TOTAL_FEATURES:
        return None, None

    feature_names = []
    groups = []

    for i in range(
        1,
        MFCC_FEATURES + 1,
    ):

        feature_names.append(
            f"MFCC {i}"
        )

        groups.append(
            "MFCC"
        )

    for i in range(
        1,
        CHROMA_FEATURES + 1,
    ):

        feature_names.append(
            f"Chroma {i}"
        )

        groups.append(
            "Chroma"
        )

    for i in range(
        1,
        MEL_FEATURES + 1,
    ):

        feature_names.append(
            f"Mel {i}"
        )

        groups.append(
            "Mel"
        )

    feature_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Group": groups,
            "Importance": importances,
        }
    )

    total_importance = (
        feature_df["Importance"].sum()
    )

    if total_importance == 0:
        return None, None

    feature_df["Importance (%)"] = (
        feature_df["Importance"]
        / total_importance
        * 100
    )

    group_df = (
        feature_df
        .groupby("Group")["Importance"]
        .sum()
        .reset_index()
    )

    group_total = (
        group_df["Importance"].sum()
    )

    group_df["Importance (%)"] = (
        group_df["Importance"]
        / group_total
        * 100
    )

    group_df = group_df.sort_values(
        "Importance (%)",
        ascending=False,
    )

    return feature_df, group_df


# ============================================================
# AUDIO STATISTICS
# ============================================================

def calculate_audio_statistics(
    audio,
    sample_rate,
):

    rms = float(
        np.sqrt(
            np.mean(
                np.square(audio)
            )
        )
    )

    peak = float(
        np.max(
            np.abs(audio)
        )
    )

    zcr = librosa.feature.zero_crossing_rate(
        audio
    )

    spectral_centroid = (
        librosa.feature.spectral_centroid(
            y=audio,
            sr=sample_rate,
        )
    )

    bandwidth = (
        librosa.feature.spectral_bandwidth(
            y=audio,
            sr=sample_rate,
        )
    )

    return {
        "RMS Energy": rms,
        "Peak Amplitude": peak,
        "Zero Crossing Rate": float(
            np.mean(zcr)
        ),
        "Spectral Centroid (Hz)": float(
            np.mean(spectral_centroid)
        ),
        "Spectral Bandwidth (Hz)": float(
            np.mean(bandwidth)
        ),
    }


# ============================================================
# TIMELINE ANALYSIS
# ============================================================

def analyze_timeline(
    file_path,
    duration,
):

    rows = []

    window = 3.0
    step = 1.5

    if duration < window:
        return pd.DataFrame(rows)

    if duration <= 3.5:

        starts = [0.0]

    else:

        starts = list(
            np.arange(
                0.5,
                duration - window + 0.01,
                step,
            )
        )

    for segment_number, start in enumerate(
        starts,
        start=1,
    ):

        try:

            result = predict_audio(
                file_path,
                offset=float(start),
            )

            rows.append(
                {
                    "Segment": segment_number,
                    "Start (sec)": round(
                        float(start),
                        2,
                    ),
                    "End (sec)": round(
                        min(
                            float(start) + window,
                            duration,
                        ),
                        2,
                    ),
                    "Emotion": result[
                        "emotion"
                    ].title(),
                    "Confidence (%)": round(
                        result[
                            "confidence"
                        ],
                        2,
                    ),
                    "Reliability": result[
                        "reliability"
                    ],
                }
            )

        except Exception as error:

            rows.append(
                {
                    "Segment": segment_number,
                    "Start (sec)": round(
                        float(start),
                        2,
                    ),
                    "End (sec)": round(
                        min(
                            float(start) + window,
                            duration,
                        ),
                        2,
                    ),
                    "Emotion": "Error",
                    "Confidence (%)": 0,
                    "Reliability": str(error),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# ADVANCED AUDIO INSIGHTS
# ============================================================

def advanced_audio_insights(
    audio,
    sample_rate,
    duration,
    quality,
):

    peak = quality["peak"]
    rms = quality["rms"]

    zcr = float(
        np.mean(
            librosa.feature.zero_crossing_rate(
                audio
            )
        )
    )

    spectral_centroid = float(
        np.mean(
            librosa.feature.spectral_centroid(
                y=audio,
                sr=sample_rate,
            )
        )
    )

    dynamic_range = float(
        peak / max(
            rms,
            1e-9,
        )
    )

    analysis_score = (
        0.80 * quality["score"]
        + 20 * min(
            1,
            duration / 5,
        )
    )

    if duration < 2:

        duration_note = (
            "Very short recording — results may be less reliable."
        )

    elif duration < 3:

        duration_note = (
            "Short recording — timeline analysis requires "
            "at least 3 seconds."
        )

    elif duration < 6:

        duration_note = (
            "Short recording — timeline analysis will contain "
            "limited segments."
        )

    else:

        duration_note = (
            "Recording duration is suitable for timeline analysis."
        )

    notes = [
        duration_note
    ]

    if quality["clipping_ratio"] > 0.01:

        notes.append(
            "Noticeable clipping detected. "
            "Consider reducing microphone volume."
        )

    if quality["silence_ratio"] > 0.60:

        notes.append(
            "A large portion of the recording is quiet or silent."
        )

    return {
        "Peak": peak,
        "RMS": rms,
        "Zero Crossing Rate": zcr,
        "Spectral Centroid": spectral_centroid,
        "Dynamic Range": dynamic_range,
        "Analysis Score": analysis_score,
        "Notes": notes,
    }



# ============================================================
# 🌟 NEW AI INSIGHT FEATURES
# ============================================================

def calculate_emotion_intensity(prediction, quality, audio, sample_rate):
    """Heuristic emotion-intensity score for presentation/analysis."""
    confidence = float(prediction["confidence"])
    margin = float(prediction["margin"])

    rms = float(np.sqrt(np.mean(np.square(audio))))
    energy_score = float(np.clip(
        (np.log10(max(rms, 1e-5)) + 3.0) / 2.5 * 100,
        0, 100
    ))

    confidence_component = np.clip(confidence, 0, 100)
    separation_component = np.clip(margin * 2.0, 0, 100)
    quality_component = np.clip(quality["score"], 0, 100)

    score = (
        0.45 * confidence_component
        + 0.20 * separation_component
        + 0.20 * energy_score
        + 0.15 * quality_component
    )
    score -= min(quality["clipping_ratio"] * 1000, 15)
    score = float(np.clip(score, 0, 100))

    if score >= 80:
        level, icon = "Very High", "🔴"
    elif score >= 65:
        level, icon = "High", "🟠"
    elif score >= 45:
        level, icon = "Moderate", "🟡"
    elif score >= 25:
        level, icon = "Low", "🟢"
    else:
        level, icon = "Very Low", "🔵"

    return {
        "score": score,
        "level": level,
        "icon": icon,
        "energy_score": float(energy_score),
        "confidence": float(confidence_component),
        "separation": float(separation_component),
    }


def detect_noise_and_health(audio, sample_rate, quality):
    """Lightweight signal-quality and noise heuristic."""
    rms = float(np.sqrt(np.mean(np.square(audio))))

    spectral_flatness = float(np.mean(
        librosa.feature.spectral_flatness(y=audio)
    ))

    zcr = float(np.mean(
        librosa.feature.zero_crossing_rate(audio)
    ))

    quiet_mask = np.abs(audio) < max(0.01, rms * 0.35)

    if np.any(quiet_mask):
        noise_floor = float(np.sqrt(
            np.mean(np.square(audio[quiet_mask]))
        ))
    else:
        noise_floor = float(rms * 0.15)

    snr_proxy = float(
        20 * np.log10(
            max(rms, 1e-7) / max(noise_floor, 1e-7)
        )
    )

    flatness_score = float(np.clip(spectral_flatness * 100, 0, 100))

    noise_score = float(np.clip(
        0.55 * flatness_score
        + 0.25 * np.clip(
            100 - max(snr_proxy, 0) * 2.5,
            0, 100
        )
        + 0.20 * np.clip(zcr * 250, 0, 100),
        0, 100
    ))

    if noise_score >= 70:
        noise_level, noise_icon = "High Noise", "🔴"
    elif noise_score >= 45:
        noise_level, noise_icon = "Moderate Noise", "🟠"
    else:
        noise_level, noise_icon = "Low Noise", "🟢"

    health_score = float(np.clip(
        0.65 * quality["score"]
        + 0.35 * (100 - noise_score),
        0, 100
    ))

    if health_score >= 85:
        health_label = "Excellent"
    elif health_score >= 70:
        health_label = "Good"
    elif health_score >= 50:
        health_label = "Fair"
    else:
        health_label = "Poor"

    recommendations = []

    if noise_score >= 45:
        recommendations.append(
            "Record in a quieter environment or move closer to the microphone."
        )

    if quality["clipping_ratio"] > 0.01:
        recommendations.append(
            "Reduce microphone gain because clipping is noticeable."
        )

    if quality["silence_ratio"] > 0.60:
        recommendations.append(
            "Reduce long silent portions and keep the speaker closer to the microphone."
        )

    if not recommendations:
        recommendations.append(
            "Recording conditions look suitable for acoustic emotion analysis."
        )

    return {
        "noise_score": noise_score,
        "noise_level": noise_level,
        "noise_icon": noise_icon,
        "health_score": health_score,
        "health_label": health_label,
        "spectral_flatness": spectral_flatness,
        "noise_floor": noise_floor,
        "snr_proxy": snr_proxy,
        "recommendations": recommendations,
    }


def create_explainability_data(prediction, audio, sample_rate, quality):
    """Builds a transparent, rule-based explanation of model output."""
    probabilities = prediction["probabilities"]

    ranked = sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True
    )

    top_emotion, top_probability = ranked[0]
    second_emotion, second_probability = ranked[1]

    stats = calculate_audio_statistics(audio, sample_rate)
    points = []

    if prediction["confidence"] >= 70:
        points.append(
            f"The model assigns a relatively strong probability to "
            f"{top_emotion.title()} ({top_probability:.1f}%)."
        )
    else:
        points.append(
            f"The model's preference for {top_emotion.title()} is not dominant "
            f"({top_probability:.1f}%), so the result should be interpreted cautiously."
        )

    points.append(
        f"The next most likely class is {second_emotion.title()} "
        f"at {second_probability:.1f}%, giving a top-2 separation of "
        f"{prediction['margin']:.1f} percentage points."
    )

    rms = stats["RMS Energy"]
    energy_note = (
        "relatively energetic" if rms >= 0.05
        else "moderate in energy" if rms >= 0.02
        else "relatively low in energy"
    )

    points.append(
        f"The recording has {energy_note} acoustic energy "
        f"(RMS {rms:.4f})."
    )

    zcr = stats["Zero Crossing Rate"]
    zcr_note = (
        "higher zero-crossing activity"
        if zcr >= 0.10
        else "lower-to-moderate zero-crossing activity"
    )

    points.append(
        f"It also shows {zcr_note} ({zcr:.4f})."
    )

    if quality["score"] < 70:
        points.append(
            "Audio quality is below the preferred range, so recording conditions "
            "may affect the prediction."
        )
    else:
        points.append(
            "The recording quality is acceptable for this acoustic analysis."
        )

    _, group_df = get_feature_importance()

    if group_df is not None and not group_df.empty:
        top_group = group_df.iloc[0]["Group"]
        group_share = group_df.iloc[0]["Importance (%)"]
        model_feature_note = (
            f"The Random Forest's largest aggregate feature group is "
            f"{top_group} ({group_share:.1f}% of model feature importance)."
        )
    else:
        model_feature_note = (
            "Model-level feature importance is not available in the loaded model."
        )

    return {
        "top_emotion": top_emotion,
        "top_probability": top_probability,
        "second_emotion": second_emotion,
        "second_probability": second_probability,
        "stats": stats,
        "points": points,
        "model_feature_note": model_feature_note,
    }


def show_emotion_radar(probabilities):
    """Display seven-class emotion probabilities as a radar chart."""
    labels = [emotion.title() for emotion in EMOTIONS]
    values = [
        float(probabilities.get(emotion, 0.0))
        for emotion in EMOTIONS
    ]

    angles = np.linspace(
        0,
        2 * np.pi,
        len(labels),
        endpoint=False
    )

    values_closed = values + values[:1]
    angles_closed = np.concatenate([angles, angles[:1]])

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)

    ax.plot(
        angles_closed,
        values_closed,
        linewidth=2
    )
    ax.fill(
        angles_closed,
        values_closed,
        alpha=0.15
    )

    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_title(
        "Emotion Probability Radar",
        pad=20
    )

    fig.tight_layout()
    show_figure(fig)
    plt.close(fig)


# ============================================================
# INTERPRETATION
# ============================================================

def create_interpretation(
    emotion,
    confidence,
    margin,
):

    description = EMOTION_DESCRIPTIONS.get(
        emotion.lower(),
        "The model detected an acoustic emotion pattern.",
    )

    if (
        confidence >= 75
        and margin >= 35
    ):

        strength = (
            "This is a very strong model prediction."
        )

    elif (
        confidence >= 55
        and margin >= 15
    ):

        strength = (
            "This is a reasonably strong model prediction."
        )

    else:

        strength = (
            "This prediction is uncertain because "
            "multiple emotions have similar probabilities."
        )

    return (
        f"{strength} "
        f"{description} "
        "This system analyzes speech acoustics and does "
        "not directly measure a person's internal emotional state."
    )


# ============================================================
# PDF REPORT
# ============================================================

def create_pdf_report(
    result,
    quality,
):

    try:

        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
        )

        buffer = BytesIO()

        document = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40,
        )

        styles = getSampleStyleSheet()

        title_style = styles["Title"]
        title_style.alignment = TA_CENTER

        story = []

        story.append(
            Paragraph(
                "SpeechSense AI",
                title_style,
            )
        )

        story.append(
            Paragraph(
                "Speech Emotion Recognition Report",
                styles["Heading2"],
            )
        )

        story.append(
            Spacer(
                1,
                0.25 * inch,
            )
        )

        report_data = [
            [
                "File",
                result["filename"],
            ],
            [
                "Predicted Emotion",
                result["emotion"].title(),
            ],
            [
                "Confidence",
                f'{result["confidence"]:.2f}%',
            ],
            [
                "Top-2 Separation",
                f'{result["margin"]:.2f}%',
            ],
            [
                "Reliability",
                result["reliability"],
            ],
            [
                "Audio Quality",
                f'{quality["score"]:.2f}% ({quality["label"]})',
            ],
            [
                "Duration",
                f'{result["duration"]:.2f} seconds',
            ],
            [
                "Sample Rate",
                f'{result["sample_rate"]} Hz',
            ],
        ]

        table = Table(
            report_data,
            colWidths=[
                2.0 * inch,
                4.0 * inch,
            ],
        )

        table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (0, -1),
                        colors.lightgrey,
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey,
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "TOP",
                    ),
                    (
                        "PADDING",
                        (0, 0),
                        (-1, -1),
                        7,
                    ),
                ]
            )
        )

        story.append(
            table
        )

        story.append(
            Spacer(
                1,
                0.3 * inch,
            )
        )

        story.append(
            Paragraph(
                "Emotion Probabilities",
                styles["Heading2"],
            )
        )

        probability_rows = [
            [
                "Emotion",
                "Probability",
            ]
        ]

        for emotion, probability in sorted(
            result["probabilities"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):

            probability_rows.append(
                [
                    emotion.title(),
                    f"{probability:.2f}%",
                ]
            )

        probability_table = Table(
            probability_rows,
            colWidths=[
                3.0 * inch,
                3.0 * inch,
            ],
        )

        probability_table.setStyle(
            TableStyle(
                [
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey,
                    ),
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.lightgrey,
                    ),
                    (
                        "PADDING",
                        (0, 0),
                        (-1, -1),
                        6,
                    ),
                ]
            )
        )

        story.append(
            probability_table
        )

        story.append(
            Spacer(
                1,
                0.3 * inch,
            )
        )

        story.append(
            Paragraph(
                "Description",
                styles["Heading2"],
            )
        )

        story.append(
            Paragraph(
                EMOTION_DESCRIPTIONS.get(
                    result["emotion"].lower(),
                    "",
                ),
                styles["BodyText"],
            )
        )

        story.append(
            Spacer(
                1,
                0.25 * inch,
            )
        )

        story.append(
            Paragraph(
                "Responsible Use",
                styles["Heading2"],
            )
        )

        story.append(
            Paragraph(
                "SpeechSense AI identifies acoustic patterns "
                "associated with emotion categories. It does "
                "not determine a person's internal emotional "
                "state and should not be used for medical, "
                "clinical, employment, legal, or other "
                "high-stakes decisions.",
                styles["BodyText"],
            )
        )

        story.append(
            Spacer(
                1,
                0.25 * inch,
            )
        )

        story.append(
            Paragraph(
                "Created by Tehmina Anwar",
                styles["BodyText"],
            )
        )

        document.build(
            story
        )

        buffer.seek(0)

        return buffer.getvalue()

    except Exception:
        return None


# ============================================================
# WAVEFORM
# ============================================================

def show_waveform(
    audio,
    sample_rate,
):

    fig, ax = plt.subplots(
        figsize=(12, 4)
    )

    librosa.display.waveshow(
        audio,
        sr=sample_rate,
        ax=ax,
    )

    ax.set_title(
        "Speech Waveform"
    )

    ax.set_xlabel(
        "Time (seconds)"
    )

    ax.set_ylabel(
        "Amplitude"
    )

    fig.tight_layout()

    show_figure(fig)

    plt.close(fig)


# ============================================================
# MEL SPECTROGRAM
# ============================================================

def show_mel_spectrogram(
    audio,
    sample_rate,
):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=128,
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    image = librosa.display.specshow(
        mel_db,
        sr=sample_rate,
        x_axis="time",
        y_axis="mel",
        ax=ax,
    )

    ax.set_title(
        "Mel Spectrogram"
    )

    fig.colorbar(
        image,
        ax=ax,
        format="%+2.0f dB",
    )

    fig.tight_layout()

    show_figure(fig)

    plt.close(fig)


# ============================================================
# MFCC VISUALIZATION
# ============================================================

def show_mfcc(
    audio,
    sample_rate,
):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40,
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    image = librosa.display.specshow(
        mfcc,
        x_axis="time",
        sr=sample_rate,
        ax=ax,
    )

    ax.set_title(
        "MFCC Features"
    )

    ax.set_ylabel(
        "MFCC Coefficients"
    )

    fig.colorbar(
        image,
        ax=ax,
    )

    fig.tight_layout()

    show_figure(fig)

    plt.close(fig)


# ============================================================
# CONFUSION MATRIX
# ============================================================

def show_confusion_matrix():

    labels = [
        emotion.title()
        for emotion in EMOTIONS
    ]

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    image = ax.imshow(
        CONFUSION_MATRIX
    )

    ax.set_xticks(
        np.arange(
            len(labels)
        )
    )

    ax.set_yticks(
        np.arange(
            len(labels)
        )
    )

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    ax.set_yticklabels(
        labels
    )

    ax.set_xlabel(
        "Predicted Label"
    )

    ax.set_ylabel(
        "True Label"
    )

    ax.set_title(
        "Confusion Matrix"
    )

    for i in range(
        CONFUSION_MATRIX.shape[0]
    ):

        for j in range(
            CONFUSION_MATRIX.shape[1]
        ):

            ax.text(
                j,
                i,
                str(
                    CONFUSION_MATRIX[i, j]
                ),
                ha="center",
                va="center",
            )

    fig.colorbar(
        image,
        ax=ax,
    )

    fig.tight_layout()

    show_figure(fig)

    plt.close(fig)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "# 🎙️ SpeechSense AI"
    )

    st.caption(
        "AI-powered Speech Emotion Recognition"
    )

    st.divider()

    st.markdown(
        "### 🎭 Supported Emotions"
    )

    for emotion in EMOTIONS:

        st.write(
            f"{EMOTION_EMOJIS[emotion]} "
            f"{emotion.title()}"
        )

    st.markdown("### 🧭 Dashboard Navigation")
    st.markdown(
        """
        <nav class="sidebar-nav" aria-label="Dashboard Navigation">
            <a href="#overview-metrics">🏠 Overview</a>
            <a href="#audio-input">🎧 Audio Input</a>
            <a href="#prediction-result">🤖 Prediction</a>
            <a href="#emotion-intensity-score">🧠 AI Insights</a>
            <a href="#batch-emotion-analysis">📦 Batch Analysis</a>
            <a href="#model-performance">📈 Model Performance</a>
            <a href="#prediction-history">🕘 History</a>
        </nav>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        "### 🤖 Model"
    )

    st.write(
        f"**{MODEL_NAME}**"
    )

    st.write(
        f"Dataset: **{DATASET_NAME}**"
    )

    st.metric(
        "Test Accuracy",
        f"{TEST_ACCURACY:.2f}%",
    )

    st.metric(
        "Training Samples",
        f"{TRAINING_SAMPLES:,}",
    )

    st.metric(
        "Testing Samples",
        f"{TESTING_SAMPLES:,}",
    )

    st.divider()

    st.markdown(
        "### 🔬 Feature Pipeline"
    )

    st.write(
        f"MFCC: **{MFCC_FEATURES}**"
    )

    st.write(
        f"Chroma: **{CHROMA_FEATURES}**"
    )

    st.write(
        f"Mel: **{MEL_FEATURES}**"
    )

    st.write(
        f"Total: **{TOTAL_FEATURES} features**"
    )

    st.divider()

    if st.button(
        "🔄 Reset Current Analysis",
        width="stretch",
    ):

        st.session_state[
            "selected_audio_path"
        ] = None

        st.session_state[
            "selected_audio_filename"
        ] = None

        st.session_state[
            "selected_audio_id"
        ] = None

        st.session_state[
            "prediction_result"
        ] = None

        st.session_state[
            "timeline_results"
        ] = None

        st.rerun()

    st.caption(
        "Created by Tehmina Anwar"
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero">

<h1>🎙️ SpeechSense AI</h1>

<p>
AI-powered Speech Emotion Recognition using
Machine Learning, Librosa and Streamlit.
</p>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# OVERVIEW METRICS
# ============================================================

st.markdown('<div id="overview-metrics"></div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "🎯 Test Accuracy",
        f"{TEST_ACCURACY:.2f}%",
    )

with col2:

    st.metric(
        "📚 Training Samples",
        f"{TRAINING_SAMPLES:,}",
    )

with col3:

    st.metric(
        "🎭 Emotions",
        len(EMOTIONS),
    )

with col4:

    st.metric(
        "🔬 Features",
        TOTAL_FEATURES,
    )


# ============================================================
# FEATURE PIPELINE
# ============================================================

st.markdown(
    "## 🔬 Feature Pipeline"
)

pipeline_cols = st.columns(3)

with pipeline_cols[0]:

    st.markdown(
        """
### 🎵 MFCC

**40 Features**

Captures important characteristics
of the speech signal and vocal tract.
"""
    )

with pipeline_cols[1]:

    st.markdown(
        """
### 🎼 Chroma

**12 Features**

Represents the distribution of
energy across musical pitch classes.
"""
    )

with pipeline_cols[2]:

    st.markdown(
        """
### 🌈 Mel Spectrogram

**40 Features**

Represents frequency content of
speech on a perceptual scale.
"""
    )


st.divider()


# ============================================================
# AUDIO INPUT
# ============================================================

st.markdown('<div id="audio-input"></div>', unsafe_allow_html=True)

st.markdown(
    "## 🎧 Select Audio"
)

input_tabs = st.tabs(
    [
        "📁 Upload Audio",
        "🎙️ Record Audio",
        "🧪 Demo Audio",
    ]
)


# ============================================================
# UPLOAD AUDIO
# ============================================================

with input_tabs[0]:

    uploaded_file = st.file_uploader(
        "Upload a speech recording",
        type=[
            "wav",
            "mp3",
            "m4a",
        ],
        help="Supported formats: WAV, MP3 and M4A",
    )

    if uploaded_file is not None:

        safe_name = safe_filename(
            uploaded_file.name
        )

        current_id = file_hash(
            uploaded_file
        )

        save_path = os.path.join(
            TEMP_FOLDER,
            f"uploaded_{current_id}_{safe_name}",
        )

        with open(
            save_path,
            "wb",
        ) as file:

            file.write(
                uploaded_file.getbuffer()
            )

        if (
            st.session_state[
                "selected_audio_id"
            ]
            != current_id
        ):

            st.session_state[
                "prediction_result"
            ] = None

            st.session_state[
                "timeline_results"
            ] = None

        st.session_state[
            "selected_audio_path"
        ] = save_path

        st.session_state[
            "selected_audio_filename"
        ] = safe_name

        st.session_state[
            "selected_audio_id"
        ] = current_id

        st.success(
            f"Selected: {safe_name}"
        )


# ============================================================
# RECORD AUDIO
# ============================================================

with input_tabs[1]:

    st.info(
        "🎙️ Record a clear speech sample using your microphone."
    )

    recorded_audio = st.audio_input(
        "Start recording"
    )

    if recorded_audio is not None:

        recording_path = os.path.join(
            TEMP_FOLDER,
            "microphone_recording.wav",
        )

        with open(
            recording_path,
            "wb",
        ) as file:

            file.write(
                recorded_audio.getbuffer()
            )

        st.session_state[
            "selected_audio_path"
        ] = recording_path

        st.session_state[
            "selected_audio_filename"
        ] = "microphone_recording.wav"

        st.session_state[
            "selected_audio_id"
        ] = "microphone"

        st.session_state[
            "prediction_result"
        ] = None

        st.session_state[
            "timeline_results"
        ] = None

        st.success(
            "Microphone recording selected."
        )


# ============================================================
# DEMO AUDIO
# ============================================================

with input_tabs[2]:

    demo_candidates = [
        os.path.join(
            TEST_AUDIO_FOLDER,
            "sample_speech.wav",
        ),
        os.path.join(
            TEST_AUDIO_FOLDER,
            "03-01-02-01-01-02-02.wav",
        ),
    ]

    available_demo = None

    for candidate in demo_candidates:

        if os.path.exists(candidate):

            available_demo = candidate
            break

    if available_demo:

        st.success(
            "Demo audio found: "
            f"{os.path.basename(available_demo)}"
        )

        if st.button(
            "🧪 Use Demo Audio",
            width="stretch",
        ):

            st.session_state[
                "selected_audio_path"
            ] = available_demo

            st.session_state[
                "selected_audio_filename"
            ] = os.path.basename(
                available_demo
            )

            st.session_state[
                "selected_audio_id"
            ] = "demo"

            st.session_state[
                "prediction_result"
            ] = None

            st.session_state[
                "timeline_results"
            ] = None

            st.rerun()

    else:

        st.warning(
            "No demo audio found."
        )

        st.code(
            """
test_audio/
└── sample_speech.wav
"""
        )


# ============================================================
# SELECTED AUDIO
# ============================================================

selected_path = st.session_state[
    "selected_audio_path"
]


if (
    selected_path
    and os.path.exists(selected_path)
):

    st.divider()

    st.markdown(
        "## 🎧 Selected Recording"
    )

    try:

        audio, sample_rate, duration = (
            get_audio_information(
                selected_path
            )
        )

        quality = calculate_audio_quality(
            audio,
            sample_rate,
            duration,
        )

    except Exception as error:

        st.error(
            f"Could not read audio: {error}"
        )

        st.stop()

    # --------------------------------------------------------
    # AUDIO PLAYER
    # --------------------------------------------------------

    with open(
        selected_path,
        "rb",
    ) as audio_file:

        audio_bytes = audio_file.read()

    extension = os.path.splitext(
        selected_path
    )[1].lower()

    if extension == ".mp3":

        audio_format = "audio/mpeg"

    elif extension == ".m4a":

        audio_format = "audio/mp4"

    else:

        audio_format = "audio/wav"

    st.audio(
        audio_bytes,
        format=audio_format,
    )

    # --------------------------------------------------------
    # AUDIO INFO
    # --------------------------------------------------------

    info1, info2, info3, info4 = st.columns(4)

    with info1:

        st.metric(
            "⏱️ Duration",
            f"{duration:.2f} sec",
        )

    with info2:

        st.metric(
            "🎚️ Sample Rate",
            f"{sample_rate:,} Hz",
        )

    with info3:

        st.metric(
            "🧪 Audio Quality",
            f'{quality["score"]:.1f}/100',
        )

    with info4:

        st.metric(
            "📌 Quality",
            quality["label"],
        )

    # --------------------------------------------------------
    # RECORDING HEALTH
    # --------------------------------------------------------

    st.markdown(
        "### 🩺 Recording Health"
    )

    health_cols = st.columns(4)

    with health_cols[0]:

        st.metric(
            "RMS",
            f'{quality["rms"]:.4f}',
        )

    with health_cols[1]:

        st.metric(
            "Peak",
            f'{quality["peak"]:.4f}',
        )

    with health_cols[2]:

        st.metric(
            "Clipping",
            f'{quality["clipping_ratio"] * 100:.2f}%',
        )

    with health_cols[3]:

        st.metric(
            "Silence",
            f'{quality["silence_ratio"] * 100:.2f}%',
        )

    if quality["score"] >= 85:

        st.success(
            "🟢 Excellent recording quality."
        )

    elif quality["score"] >= 70:

        st.info(
            "🟡 Good recording quality."
        )

    elif quality["score"] >= 50:

        st.warning(
            "🟠 Fair recording quality. "
            "A cleaner recording may improve reliability."
        )

    else:

        st.error(
            "🔴 Poor recording quality. "
            "Consider recording again in a quieter environment."
        )

    # --------------------------------------------------------
    # AUDIO VISUALIZATION
    # --------------------------------------------------------

    visual_tabs = st.tabs(
        [
            "🌊 Waveform",
            "🌈 Mel Spectrogram",
            "🎼 MFCC",
            "📊 Acoustic Statistics",
        ]
    )

    with visual_tabs[0]:

        show_waveform(
            audio,
            sample_rate,
        )

    with visual_tabs[1]:

        show_mel_spectrogram(
            audio,
            sample_rate,
        )

    with visual_tabs[2]:

        show_mfcc(
            audio,
            sample_rate,
        )

    with visual_tabs[3]:

        stats = calculate_audio_statistics(
            audio,
            sample_rate,
        )

        stats_df = pd.DataFrame(
            {
                "Metric": list(
                    stats.keys()
                ),
                "Value": [
                    round(
                        value,
                        5,
                    )
                    for value in stats.values()
                ],
            }
        )

        st.dataframe(
            stats_df,
            width="stretch",
            hide_index=True,
        )

    # --------------------------------------------------------
    # ADVANCED INSIGHTS
    # --------------------------------------------------------

    with st.expander(
        "🧠 Advanced Audio Insights",
        expanded=False,
    ):

        insights = advanced_audio_insights(
            audio,
            sample_rate,
            duration,
            quality,
        )

        insight_cols = st.columns(3)

        with insight_cols[0]:

            st.metric(
                "Peak",
                f'{insights["Peak"]:.4f}',
            )

            st.metric(
                "RMS",
                f'{insights["RMS"]:.4f}',
            )

        with insight_cols[1]:

            st.metric(
                "Zero Crossing Rate",
                f'{insights["Zero Crossing Rate"]:.4f}',
            )

            st.metric(
                "Spectral Centroid",
                f'{insights["Spectral Centroid"]:.1f} Hz',
            )

        with insight_cols[2]:

            st.metric(
                "Dynamic Range",
                f'{insights["Dynamic Range"]:.2f}',
            )

            st.metric(
                "Analysis Score",
                f'{insights["Analysis Score"]:.1f}/100',
            )

        for note in insights["Notes"]:

            st.info(note)

    # ========================================================
    # PREDICTION
    # ========================================================

    st.divider()

    st.markdown(
        "## 🤖 Emotion Prediction"
    )

    predict_col1, predict_col2 = st.columns(
        [3, 1]
    )

    with predict_col1:

        st.info(
            "The model analyzes up to 3 seconds "
            "starting from a 0.5-second offset."
        )

    with predict_col2:

        predict_button = st.button(
            "🔮 Predict Emotion",
            type="primary",
            width="stretch",
        )

    if predict_button:

        if duration < 1:

            st.error(
                "Audio is too short. "
                "Please provide at least 1 second of audio."
            )

        else:

            try:

                with st.spinner(
                    "🎙️ Extracting audio features..."
                ):

                    prediction = predict_audio(
                        selected_path
                    )

                recommendation, recommendation_text = (
                    smart_recommendation(
                        prediction["confidence"],
                        prediction["margin"],
                        quality["score"],
                    )
                )

                result_data = {
                    **prediction,
                    "filename": st.session_state[
                        "selected_audio_filename"
                    ],
                    "timestamp": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "sample_rate": sample_rate,
                    "duration": duration,
                    "quality": quality,
                    "recommendation": recommendation,
                    "recommendation_text": recommendation_text,
                }

                st.session_state[
                    "prediction_result"
                ] = result_data

                # Add to history
                st.session_state[
                    "history"
                ].append(
                    {
                        "Time": result_data[
                            "timestamp"
                        ],
                        "File": result_data[
                            "filename"
                        ],
                        "Emotion": result_data[
                            "emotion"
                        ].title(),
                        "Confidence (%)": round(
                            result_data[
                                "confidence"
                            ],
                            2,
                        ),
                        "Top-2 Gap (%)": round(
                            result_data[
                                "margin"
                            ],
                            2,
                        ),
                        "Reliability": result_data[
                            "reliability"
                        ],
                        "Quality (%)": round(
                            quality["score"],
                            2,
                        ),
                        "Duration (sec)": round(
                            duration,
                            2,
                        ),
                    }
                )

                st.success(
                    "✅ Prediction completed successfully."
                )

            except Exception as error:

                st.error(
                    f"Prediction failed: {error}"
                )

    # ========================================================
    # RESULT DASHBOARD
    # ========================================================

    result = st.session_state[
        "prediction_result"
    ]

    if result is not None:

        st.divider()

        st.markdown('<div id="prediction-result"></div>', unsafe_allow_html=True)

        st.markdown(
            "## 🎯 Prediction Result"
        )

        emotion = result[
            "emotion"
        ]

        confidence = result[
            "confidence"
        ]

        margin = result[
            "margin"
        ]

        emoji = EMOTION_EMOJIS.get(
            emotion.lower(),
            "🎭",
        )

        st.markdown(
            f"""
<div class="big-result">

<div class="big-emotion">
{emoji} {emotion.title()}
</div>

<div class="confidence-number">
{confidence:.2f}%
</div>

<p>Model Confidence</p>

</div>
""",
            unsafe_allow_html=True,
        )

        st.write("")

        result_cols = st.columns(3)

        with result_cols[0]:

            st.metric(
                "🎯 Confidence",
                f"{confidence:.2f}%",
            )

        with result_cols[1]:

            st.metric(
                "📏 Top-2 Separation",
                f"{margin:.2f}%",
            )

        with result_cols[2]:

            st.metric(
                "🧠 Reliability",
                result[
                    "reliability"
                ],
            )

        # ----------------------------------------------------
        # RELIABILITY
        # ----------------------------------------------------

        st.markdown(
            "### "
            f'{result["reliability_icon"]} '
            f'{result["reliability"]}'
        )

        st.info(
            result[
                "reliability_text"
            ]
        )

        # ----------------------------------------------------
        # EMOTION ANALYSIS
        # ----------------------------------------------------

        st.markdown(
            "### 🎭 Emotion Analysis"
        )

        st.write(
            EMOTION_DESCRIPTIONS.get(
                emotion.lower(),
                "",
            )
        )

        # ----------------------------------------------------
        # SMART RECOMMENDATION
        # ----------------------------------------------------

        st.markdown(
            "### 💡 Smart Recommendation"
        )

        recommendation = result[
            "recommendation"
        ]

        if recommendation == "Strong Result":

            st.success(
                f"🟢 **{recommendation}**\n\n"
                + result[
                    "recommendation_text"
                ]
            )

        elif recommendation == "Strong Prediction / Improve Audio":

            st.warning(
                f"🟡 **{recommendation}**\n\n"
                + result[
                    "recommendation_text"
                ]
            )

        elif recommendation == "Moderate Result":

            st.info(
                f"🟡 **{recommendation}**\n\n"
                + result[
                    "recommendation_text"
                ]
            )

        else:

            st.error(
                f"🔴 **{recommendation}**\n\n"
                + result[
                    "recommendation_text"
                ]
            )

        # ----------------------------------------------------
        # INTERPRETATION
        # ----------------------------------------------------

        st.markdown(
            "### 📝 Interpretation"
        )

        st.write(
            create_interpretation(
                emotion,
                confidence,
                margin,
            )
        )

        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        st.markdown(
            "### 📊 Emotion Probabilities"
        )

        probability_df = pd.DataFrame(
            {
                "Emotion": [
                    key.title()
                    for key in result[
                        "probabilities"
                    ].keys()
                ],
                "Probability (%)": [
                    round(
                        value,
                        2,
                    )
                    for value in result[
                        "probabilities"
                    ].values()
                ],
            }
        )

        probability_df = (
            probability_df
            .sort_values(
                "Probability (%)",
                ascending=False,
            )
        )

        prob_col1, prob_col2 = st.columns(
            [1, 2]
        )

        with prob_col1:

            st.dataframe(
                probability_df,
                width="stretch",
                hide_index=True,
            )

        with prob_col2:

            chart_df = probability_df.set_index("Emotion")

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(chart_df.index, chart_df["Probability (%)"])
            ax.set_ylabel("Probability (%)")
            ax.set_xlabel("Emotion")
            ax.set_ylim(0, max(100, float(chart_df["Probability (%)"].max()) * 1.15))
            ax.set_title("Emotion Probability Distribution")
            ax.tick_params(axis="x", rotation=25)
            fig.tight_layout()
            show_figure(fig)

        # ----------------------------------------------------
        # TOP 3
        # ----------------------------------------------------

        st.markdown(
            "### 🏆 Top 3 Predictions"
        )

        top3 = probability_df.head(3)

        top_cols = st.columns(3)

        for position, (_, row) in enumerate(
            top3.iterrows()
        ):

            with top_cols[position]:

                emotion_name = row[
                    "Emotion"
                ]

                emotion_key = (
                    emotion_name.lower()
                )

                st.markdown(
                    f"""
### {position + 1}.
{EMOTION_EMOJIS.get(emotion_key, "🎭")}

**{emotion_name}**

### {row["Probability (%)"]:.2f}%
"""
                )

        # ----------------------------------------------------
        # FEATURE IMPORTANCE
        # ----------------------------------------------------

        st.markdown(
            "### 🔬 Feature Importance"
        )

        feature_df, group_df = (
            get_feature_importance()
        )

        if group_df is not None:

            st.write(
                "Feature importance shows which "
                "audio feature groups contribute "
                "most to the Random Forest model."
            )

            st.dataframe(
                group_df.round(4),
                width="stretch",
                hide_index=True,
            )

            importance_chart = group_df.set_index("Group")
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(importance_chart.index, importance_chart["Importance (%)"])
            ax.set_ylabel("Importance (%)")
            ax.set_xlabel("Feature Group")
            ax.set_title("Aggregate Feature Importance")
            ax.tick_params(axis="x", rotation=20)
            fig.tight_layout()
            show_figure(fig)

            with st.expander(
                "View Individual Feature Importance"
            ):

                st.dataframe(
                    feature_df.sort_values(
                        "Importance (%)",
                        ascending=False,
                    ).round(4),
                    width="stretch",
                    hide_index=True,
                )

        else:

            st.info(
                "Feature importance is not available "
                "for the loaded model."
            )

        # ----------------------------------------------------
        # TIMELINE
        # ----------------------------------------------------

        st.markdown(
            "### ⏱️ Emotion Timeline"
        )

        if duration < 3.0:

            st.warning(
                "Timeline analysis requires at least "
                "3 seconds of audio."
            )

        else:

            timeline_button = st.button(
                "📈 Analyze Emotion Over Time",
                width="stretch",
            )

            if timeline_button:

                with st.spinner(
                    "Analyzing audio segments..."
                ):

                    timeline_df = analyze_timeline(
                        selected_path,
                        duration,
                    )

                st.session_state[
                    "timeline_results"
                ] = timeline_df

            timeline_df = st.session_state[
                "timeline_results"
            ]

            if (
                timeline_df is not None
                and not timeline_df.empty
            ):

                st.dataframe(
                    timeline_df,
                    width="stretch",
                    hide_index=True,
                )

                valid_timeline = timeline_df[
                    timeline_df["Emotion"] != "Error"
                ]

                if not valid_timeline.empty:

                    timeline_chart = (
                        valid_timeline[
                            [
                                "Start (sec)",
                                "Confidence (%)",
                            ]
                        ]
                        .set_index(
                            "Start (sec)"
                        )
                    )

                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(
                        timeline_chart.index,
                        timeline_chart["Confidence (%)"],
                        marker="o",
                    )
                    ax.set_xlabel("Start (sec)")
                    ax.set_ylabel("Confidence (%)")
                    ax.set_title("Emotion Confidence Timeline")
                    ax.set_ylim(0, 100)
                    fig.tight_layout()
                    show_figure(fig)

            elif (
                timeline_df is not None
                and timeline_df.empty
            ):

                st.info(
                    "No timeline segments were generated."
                )

        # ----------------------------------------------------
        # FEATURE EXPLORER
        # ----------------------------------------------------

        with st.expander(
            "🔍 Feature Explorer"
        ):

            try:

                extracted_features = (
                    extract_features(
                        selected_path
                    )
                )

                feature_explorer = pd.DataFrame(
                    {
                        "Feature Index": range(
                            1,
                            len(
                                extracted_features
                            ) + 1,
                        ),
                        "Value": extracted_features,
                    }
                )

                st.dataframe(
                    feature_explorer,
                    width="stretch",
                    hide_index=True,
                )

            except Exception as error:

                st.error(
                    f"Feature explorer failed: {error}"
                )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        with st.expander(
            "📋 Analysis Summary",
            expanded=False,
        ):

            summary_df = pd.DataFrame(
                {
                    "Property": [
                        "File",
                        "Emotion",
                        "Confidence",
                        "Top-2 Separation",
                        "Reliability",
                        "Quality",
                        "Duration",
                        "Sample Rate",
                        "Recommendation",
                    ],
                    "Value": [
                        result["filename"],
                        emotion.title(),
                        f"{confidence:.2f}%",
                        f"{margin:.2f}%",
                        result["reliability"],
                        f'{quality["score"]:.2f}%',
                        f"{duration:.2f} sec",
                        f"{sample_rate:,} Hz",
                        recommendation,
                    ],
                }
            )

            st.dataframe(
                summary_df,
                width="stretch",
                hide_index=True,
            )


        # ----------------------------------------------------
        # 🌡️ EMOTION INTENSITY SCORE
        # ----------------------------------------------------

        st.markdown('<div id="emotion-intensity-score"></div>', unsafe_allow_html=True)

        st.markdown(
            "### 🌡️ Emotion Intensity Score"
        )

        intensity = calculate_emotion_intensity(
            result,
            quality,
            audio,
            sample_rate,
        )

        intensity_cols = st.columns(4)

        with intensity_cols[0]:
            st.metric(
                "Intensity Score",
                f'{intensity["score"]:.1f}/100',
            )

        with intensity_cols[1]:
            st.metric(
                "Intensity Level",
                f'{intensity["icon"]} {intensity["level"]}',
            )

        with intensity_cols[2]:
            st.metric(
                "Acoustic Energy",
                f'{intensity["energy_score"]:.1f}/100',
            )

        with intensity_cols[3]:
            st.metric(
                "Prediction Strength",
                f'{intensity["confidence"]:.1f}/100',
            )

        st.caption(
            "This is a heuristic interpretation score based on model confidence, "
            "probability separation, acoustic energy and recording quality. "
            "It does not measure a person's true emotional intensity."
        )

        # ----------------------------------------------------
        # 🧠 EXPLAINABLE AI
        # ----------------------------------------------------

        st.markdown(
            "### 🧠 Explainable AI — Why this emotion?"
        )

        explainability = create_explainability_data(
            result,
            audio,
            sample_rate,
            quality,
        )

        explain_col1, explain_col2 = st.columns(2)

        with explain_col1:
            st.markdown(
                f"""
                **Predicted emotion:** {EMOTION_EMOJIS.get(
                    explainability["top_emotion"],
                    "🎭"
                )} {explainability["top_emotion"].title()}

                **Model probability:** {explainability["top_probability"]:.2f}%

                **Runner-up:** {explainability["second_emotion"].title()}
                ({explainability["second_probability"]:.2f}%)

                **Top-2 separation:** {result["margin"]:.2f} percentage points
                """
            )

        with explain_col2:
            for point in explainability["points"]:
                st.info(point)

        st.caption(
            explainability["model_feature_note"]
        )

        st.warning(
            "Explainability describes model output and measurable acoustic "
            "properties. It is not proof of a person's internal emotional state."
        )

        # ----------------------------------------------------
        # 📊 EMOTION RADAR CHART
        # ----------------------------------------------------

        st.markdown(
            "### 📊 Emotion Radar Chart"
        )

        radar_col1, radar_col2 = st.columns(
            [1.35, 1]
        )

        with radar_col1:
            show_emotion_radar(
                result["probabilities"]
            )

        with radar_col2:
            st.markdown(
                "#### Probability Profile"
            )

            radar_df = pd.DataFrame(
                {
                    "Emotion": [
                        emotion.title()
                        for emotion in EMOTIONS
                    ],
                    "Probability (%)": [
                        round(
                            result["probabilities"].get(
                                emotion,
                                0.0,
                            ),
                            2,
                        )
                        for emotion in EMOTIONS
                    ],
                }
            )

            st.dataframe(
                radar_df.sort_values(
                    "Probability (%)",
                    ascending=False,
                ),
                width="stretch",
                hide_index=True,
            )

            st.caption(
                "The radar visualizes the model probability distribution "
                "across all seven supported emotion classes."
            )

        # ----------------------------------------------------
        # 🔊 NOISE DETECTION + AUDIO HEALTH
        # ----------------------------------------------------

        st.markdown(
            "### 🔊 Noise Detection + Audio Health Score"
        )

        noise_health = detect_noise_and_health(
            audio,
            sample_rate,
            quality,
        )

        health_cols = st.columns(4)

        with health_cols[0]:
            st.metric(
                "Audio Health",
                f'{noise_health["health_score"]:.1f}/100',
            )

        with health_cols[1]:
            st.metric(
                "Health Level",
                noise_health["health_label"],
            )

        with health_cols[2]:
            st.metric(
                "Noise Level",
                f'{noise_health["noise_icon"]} {noise_health["noise_level"]}',
            )

        with health_cols[3]:
            st.metric(
                "SNR Proxy",
                f'{noise_health["snr_proxy"]:.1f} dB',
            )

        nh_col1, nh_col2 = st.columns(2)

        with nh_col1:
            st.markdown(
                "#### 🔬 Signal Health Details"
            )

            noise_df = pd.DataFrame(
                {
                    "Metric": [
                        "Audio Quality",
                        "Noise Score",
                        "Spectral Flatness",
                        "Noise Floor Proxy",
                        "SNR Proxy",
                        "Silence Ratio",
                        "Clipping Ratio",
                    ],
                    "Value": [
                        f'{quality["score"]:.1f}/100',
                        f'{noise_health["noise_score"]:.1f}/100',
                        f'{noise_health["spectral_flatness"]:.5f}',
                        f'{noise_health["noise_floor"]:.6f}',
                        f'{noise_health["snr_proxy"]:.1f} dB',
                        f'{quality["silence_ratio"] * 100:.1f}%',
                        f'{quality["clipping_ratio"] * 100:.2f}%',
                    ],
                }
            )

            st.dataframe(
                noise_df,
                width="stretch",
                hide_index=True,
            )

        with nh_col2:
            st.markdown(
                "#### 💡 Recording Recommendations"
            )

            for note in noise_health["recommendations"]:
                if noise_health["noise_score"] >= 70:
                    st.error(f"🔴 {note}")
                elif noise_health["noise_score"] >= 45:
                    st.warning(f"🟠 {note}")
                else:
                    st.success(f"🟢 {note}")

        st.caption(
            "Noise detection uses signal-level heuristics including spectral "
            "flatness, quiet-region energy and zero-crossing activity. "
            "SNR is a proxy rather than a calibrated laboratory measurement."
        )

        # ----------------------------------------------------
        # EXPORT
        # ----------------------------------------------------

        st.markdown(
            "### 📥 Export Results"
        )

        export_col1, export_col2 = st.columns(2)

        with export_col1:

            csv_data = probability_df.to_csv(
                index=False
            )

            st.download_button(
                "📊 Download Prediction CSV",
                data=csv_data,
                file_name="speech_emotion_prediction.csv",
                mime="text/csv",
                width="stretch",
            )

        with export_col2:

            pdf_data = create_pdf_report(
                result,
                quality,
            )

            if pdf_data:

                st.download_button(
                    "📄 Download PDF Report",
                    data=pdf_data,
                    file_name="SpeechSense_AI_Report.pdf",
                    mime="application/pdf",
                    width="stretch",
                )

            else:

                st.warning(
                    "PDF generation unavailable. "
                    "Install reportlab."
                )


# ============================================================
# BATCH ANALYSIS
# ============================================================

st.divider()

st.markdown('<div id="batch-emotion-analysis"></div>', unsafe_allow_html=True)

st.markdown(
    "## 📦 Batch Emotion Analysis"
)

batch_files = st.file_uploader(
    "Upload multiple speech recordings",
    type=[
        "wav",
        "mp3",
        "m4a",
    ],
    accept_multiple_files=True,
    key="batch_uploader",
)

if batch_files:

    if st.button(
        "🚀 Analyze All Files",
        width="stretch",
    ):

        batch_rows = []

        progress = st.progress(
            0
        )

        status_text = st.empty()

        total_files = len(
            batch_files
        )

        for index, batch_file in enumerate(
            batch_files
        ):

            batch_name = safe_filename(
                batch_file.name
            )

            batch_path = os.path.join(
                TEMP_FOLDER,
                f"batch_{index}_{batch_name}",
            )

            status_text.write(
                f"Analyzing {index + 1}/{total_files}: "
                f"{batch_name}"
            )

            try:

                with open(
                    batch_path,
                    "wb",
                ) as file:

                    file.write(
                        batch_file.getbuffer()
                    )

                audio, sr, duration = (
                    get_audio_information(
                        batch_path
                    )
                )

                quality = calculate_audio_quality(
                    audio,
                    sr,
                    duration,
                )

                if duration < 1:

                    raise ValueError(
                        "Audio is shorter than 1 second."
                    )

                prediction = predict_audio(
                    batch_path
                )

                recommendation, _ = (
                    smart_recommendation(
                        prediction["confidence"],
                        prediction["margin"],
                        quality["score"],
                    )
                )

                batch_rows.append(
                    {
                        "File": batch_name,
                        "Emotion": prediction[
                            "emotion"
                        ].title(),
                        "Confidence (%)": round(
                            prediction[
                                "confidence"
                            ],
                            2,
                        ),
                        "Top-2 Gap (%)": round(
                            prediction[
                                "margin"
                            ],
                            2,
                        ),
                        "Reliability": prediction[
                            "reliability"
                        ],
                        "Quality": round(
                            quality["score"],
                            2,
                        ),
                        "Duration": round(
                            duration,
                            2,
                        ),
                        "Recommendation": recommendation,
                    }
                )

            except Exception as error:

                batch_rows.append(
                    {
                        "File": batch_name,
                        "Emotion": "Error",
                        "Confidence (%)": 0,
                        "Top-2 Gap (%)": 0,
                        "Reliability": "Error",
                        "Quality": 0,
                        "Duration": 0,
                        "Recommendation": str(
                            error
                        ),
                    }
                )

            progress.progress(
                (index + 1)
                / total_files
            )

        status_text.empty()

        st.session_state[
            "batch_results"
        ] = pd.DataFrame(
            batch_rows
        )

        st.success(
            f"✅ Completed analysis of "
            f"{total_files} files."
        )


# ============================================================
# DISPLAY BATCH RESULTS
# ============================================================

batch_df = st.session_state[
    "batch_results"
]

if (
    batch_df is not None
    and not batch_df.empty
):

    st.markdown(
        "### 📊 Batch Results"
    )

    st.dataframe(
        batch_df,
        width="stretch",
        hide_index=True,
    )

    valid_batch = batch_df[
        batch_df["Emotion"] != "Error"
    ].copy()

    if not valid_batch.empty:

        batch_metrics = st.columns(5)

        with batch_metrics[0]:

            st.metric(
                "📁 Files Analyzed",
                len(valid_batch),
            )

        with batch_metrics[1]:

            st.metric(
                "🎯 Avg Confidence",
                f'{valid_batch["Confidence (%)"].mean():.2f}%',
            )

        with batch_metrics[2]:

            most_common = (
                valid_batch[
                    "Emotion"
                ]
                .value_counts()
                .idxmax()
            )

            st.metric(
                "🎭 Most Common",
                most_common,
            )

        with batch_metrics[3]:

            st.metric(
                "🎚️ Avg Quality",
                f'{valid_batch["Quality"].mean():.2f}%',
            )

        with batch_metrics[4]:

            st.metric(
                "📏 Avg Gap",
                f'{valid_batch["Top-2 Gap (%)"].mean():.2f}%',
            )

        batch_tab1, batch_tab2, batch_tab3 = st.tabs(
            [
                "🎭 Emotion Distribution",
                "🎯 Confidence",
                "🎚️ Quality",
            ]
        )

        with batch_tab1:

            emotion_counts = (
                valid_batch[
                    "Emotion"
                ]
                .value_counts()
            )

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(emotion_counts.index, emotion_counts.values)
            ax.set_xlabel("Emotion")
            ax.set_ylabel("Number of Files")
            ax.set_title("Batch Emotion Distribution")
            ax.tick_params(axis="x", rotation=25)
            fig.tight_layout()
            show_figure(fig)

        with batch_tab2:

            confidence_chart = (
                valid_batch[
                    [
                        "File",
                        "Confidence (%)",
                    ]
                ]
                .set_index(
                    "File"
                )
            )

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(confidence_chart.index, confidence_chart["Confidence (%)"])
            ax.set_xlabel("File")
            ax.set_ylabel("Confidence (%)")
            ax.set_title("Batch Prediction Confidence")
            ax.set_ylim(0, 100)
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            show_figure(fig)

        with batch_tab3:

            quality_chart = (
                valid_batch[
                    [
                        "File",
                        "Quality",
                    ]
                ]
                .set_index(
                    "File"
                )
            )

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(quality_chart.index, quality_chart["Quality"])
            ax.set_xlabel("File")
            ax.set_ylabel("Quality Score")
            ax.set_title("Batch Recording Quality")
            ax.set_ylim(0, 100)
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            show_figure(fig)

        # ----------------------------------------------------
        # BATCH HIGHLIGHTS
        # ----------------------------------------------------

        st.markdown(
            "### 🏆 Batch Highlights"
        )

        best_cols = st.columns(3)

        best_confidence = valid_batch.loc[
            valid_batch[
                "Confidence (%)"
            ].idxmax()
        ]

        best_quality = valid_batch.loc[
            valid_batch[
                "Quality"
            ].idxmax()
        ]

        best_gap = valid_batch.loc[
            valid_batch[
                "Top-2 Gap (%)"
            ].idxmax()
        ]

        with best_cols[0]:

            st.success(
                f'**Best Confidence**\n\n'
                f'{best_confidence["File"]}\n\n'
                f'{best_confidence["Confidence (%)"]:.2f}%'
            )

        with best_cols[1]:

            st.info(
                f'**Best Quality**\n\n'
                f'{best_quality["File"]}\n\n'
                f'{best_quality["Quality"]:.2f}%'
            )

        with best_cols[2]:

            st.warning(
                f'**Best Separation**\n\n'
                f'{best_gap["File"]}\n\n'
                f'{best_gap["Top-2 Gap (%)"]:.2f}%'
            )

        batch_csv = batch_df.to_csv(
            index=False
        )

        st.download_button(
            "📥 Download Batch Results CSV",
            data=batch_csv,
            file_name="speechsense_batch_results.csv",
            mime="text/csv",
            width="stretch",
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.markdown('<div id="model-performance"></div>', unsafe_allow_html=True)

with st.expander(
    "📈 Model Performance",
    expanded=False,
):

    performance_cols = st.columns(4)

    with performance_cols[0]:

        st.metric(
            "Accuracy",
            f"{TEST_ACCURACY:.2f}%",
        )

    with performance_cols[1]:

        st.metric(
            "Training",
            f"{TRAINING_SAMPLES:,}",
        )

    with performance_cols[2]:

        st.metric(
            "Testing",
            f"{TESTING_SAMPLES:,}",
        )

    with performance_cols[3]:

        st.metric(
            "Features",
            TOTAL_FEATURES,
        )

    st.markdown(
        "### Classification Report"
    )

    display_report = (
        CLASSIFICATION_REPORT.copy()
    )

    for column in [
        "Precision",
        "Recall",
        "F1-Score",
    ]:

        display_report[column] = (
            display_report[column]
            * 100
        ).round(2).astype(str) + "%"

    st.dataframe(
        display_report,
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        "### Confusion Matrix"
    )

    show_confusion_matrix()


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

with st.expander(
    "🤖 Model Information",
    expanded=False,
):

    model_info_cols = st.columns(2)

    with model_info_cols[0]:

        st.markdown(
            f"""
### Model

**{MODEL_NAME}**

### Dataset

**{DATASET_NAME}**

### Evaluation Accuracy

**{TEST_ACCURACY:.2f}%**

### Training Samples

**{TRAINING_SAMPLES:,}**

### Testing Samples

**{TESTING_SAMPLES:,}**
"""
        )

    with model_info_cols[1]:

        st.markdown(
            """
### Feature Composition

| Feature | Count |
|---|---:|
| MFCC | 40 |
| Chroma | 12 |
| Mel | 40 |
| **Total** | **92** |

### Technical Pipeline

Audio

↓

Preprocessing

↓

40 MFCC + 12 Chroma + 40 Mel

↓

92-dimensional Feature Vector

↓

StandardScaler

↓

Random Forest

↓

Probability Distribution

↓

Emotion Prediction
"""
        )


# ============================================================
# RESPONSIBLE AI
# ============================================================

st.divider()

with st.expander(
    "🛡️ Responsible AI & Limitations",
    expanded=False,
):

    st.markdown(
        """
### Important Information

SpeechSense AI identifies **acoustic patterns**
associated with emotion categories.

It does **not** directly measure a person's
internal emotional state.

### Factors That Can Affect Results

- 🎙️ Microphone quality
- 🔊 Background noise
- 🗣️ Speaker characteristics
- 🌍 Accent
- 🗣️ Language
- ⏱️ Recording duration
- 🏠 Recording environment
- 🎚️ Volume and clipping

### Accuracy Disclaimer

The reported **94.10% accuracy** is an evaluation
result on the project's test data.

It does not guarantee the same performance on
new real-world recordings.

### Intended Use

This project is intended for:

- Educational demonstrations
- Machine learning research
- Speech analysis experiments
- AI/ML portfolio demonstration

It should **not** be used for:

- Medical diagnosis
- Mental health diagnosis
- Employment decisions
- Legal decisions
- High-stakes profiling

The Smart Recommendation feature is only an
additional interpretation aid.
"""
    )


# ============================================================
# HISTORY
# ============================================================

st.divider()

st.markdown('<div id="prediction-history"></div>', unsafe_allow_html=True)

with st.expander(
    "🕘 Prediction History",
    expanded=False,
):

    history = st.session_state[
        "history"
    ]

    if history:

        history_df = pd.DataFrame(
            history
        )

        st.dataframe(
            history_df,
            width="stretch",
            hide_index=True,
        )

        history_csv = history_df.to_csv(
            index=False
        )

        st.download_button(
            "📥 Download History CSV",
            data=history_csv,
            file_name="speechsense_prediction_history.csv",
            mime="text/csv",
            width="stretch",
        )

        if st.button(
            "🗑️ Clear History",
            width="stretch",
        ):

            st.session_state[
                "history"
            ] = []

            st.rerun()

    else:

        st.info(
            "No predictions have been recorded yet."
        )


# ============================================================
# HISTORY ANALYTICS
# ============================================================

if st.session_state[
    "history"
]:

    history_df = pd.DataFrame(
        st.session_state[
            "history"
        ]
    )

    st.divider()

    with st.expander(
        "📊 History Analytics",
        expanded=False,
    ):

        analytics_cols = st.columns(4)

        with analytics_cols[0]:

            st.metric(
                "Total Predictions",
                len(history_df),
            )

        with analytics_cols[1]:

            st.metric(
                "Avg Confidence",
                f'{history_df["Confidence (%)"].mean():.2f}%',
            )

        with analytics_cols[2]:

            st.metric(
                "Avg Quality",
                f'{history_df["Quality (%)"].mean():.2f}%',
            )

        with analytics_cols[3]:

            st.metric(
                "Best Confidence",
                f'{history_df["Confidence (%)"].max():.2f}%',
            )

        st.markdown(
            "### Emotion Distribution"
        )

        emotion_history = (
            history_df[
                "Emotion"
            ]
            .value_counts()
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(emotion_history.index.astype(str), emotion_history.values)
        ax.set_title("Emotion Distribution")
        ax.set_xlabel("Emotion")
        ax.set_ylabel("Predictions")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        show_figure(fig)

        st.markdown(
            "### Confidence Over Time"
        )

        confidence_history = (
            history_df[
                [
                    "Time",
                    "Confidence (%)",
                ]
            ]
            .set_index(
                "Time"
            )
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(confidence_history) + 1), confidence_history["Confidence (%)"].values, marker="o")
        ax.set_title("Confidence Over Time")
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Confidence (%)")
        ax.set_ylim(0, 100)
        fig.tight_layout()
        show_figure(fig)

        st.markdown(
            "### Recording Quality Over Time"
        )

        quality_history = (
            history_df[
                [
                    "Time",
                    "Quality (%)",
                ]
            ]
            .set_index(
                "Time"
            )
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(quality_history) + 1), quality_history["Quality (%)"].values, marker="o")
        ax.set_title("Recording Quality Over Time")
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Quality (%)")
        ax.set_ylim(0, 100)
        fig.tight_layout()
        show_figure(fig)


# ============================================================
# HOW TO USE
# ============================================================

st.divider()

with st.expander(
    "📖 How to Use SpeechSense AI",
    expanded=False,
):

    st.markdown(
        """
### 1️⃣ Select Audio

You can upload a WAV/MP3/M4A file, record directly using your microphone, or use the demo recording.

### 2️⃣ Record Clearly

For better results, speak close to the microphone, avoid background noise and excessive volume, speak naturally, and use at least 3 seconds for timeline analysis.

### 3️⃣ Check Recording Health

Review:

- Duration
- Sample rate
- RMS
- Peak
- Clipping
- Silence ratio
- Overall quality score

### 4️⃣ Explore Audio

SpeechSense AI provides waveform, Mel Spectrogram, MFCC visualization, acoustic statistics, and advanced audio insights.

### 5️⃣ Predict Emotion

Click:

**🔮 Predict Emotion**

The Random Forest model generates:

Predicted emotion, confidence, probability distribution, top-2 separation, and reliability level.

### 6️⃣ Review Smart Recommendation

The application evaluates:

Confidence, prediction separation, and audio quality.

and provides an additional interpretation.

### 7️⃣ Feature Importance

The application shows the contribution of:

MFCC, Chroma, and Mel features.

when model feature importance is available.

### 8️⃣ Timeline Analysis

For recordings of at least 3 seconds,
the application can analyze different
speech segments over time.

### 9️⃣ Batch Analysis

Upload multiple recordings to compare:

Emotions, confidence, quality, top-2 gap, reliability, and recommendations.

### 🔟 Explainable & Audio Health Features

The application also provides:

Emotion Intensity Score, Explainable AI interpretation, emotion probability radar, Noise Detection, and Audio Health Score.

These are acoustic/model-output interpretation aids and are not clinical measurements.

### 1️⃣1️⃣ Export

You can export:

Prediction CSV, Batch CSV, History CSV, and PDF report.
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">

<hr>

<h3>🎙️ SpeechSense AI</h3>

<p>
AI-powered Speech Emotion Recognition
</p>

<p>
Python • Librosa • Scikit-learn • Streamlit
</p>

<p>
Random Forest Classifier • 94.10% Evaluation Accuracy
</p>

<p>
Created by <strong>Tehmina Anwar</strong>
</p>

</div>
""",
    unsafe_allow_html=True,
)