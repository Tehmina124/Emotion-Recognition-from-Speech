import os
import shutil

# ============================================================
# 🎙️ RAVDESS Dataset Organizer
# ============================================================

# RAVDESS Actor folders are inside the "dataset" folder
SOURCE_FOLDER = "dataset"

# Organized emotion files will be saved here
DEST_FOLDER = "emotion_data"

# RAVDESS emotion codes
EMOTIONS = {
    "01": "neutral",
    "02": "neutral",   # calm ko neutral mein combine kar rahe hain
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise"
}


def organize_dataset():

    # Check source folder
    if not os.path.exists(SOURCE_FOLDER):
        print("❌ 'dataset' folder nahi mila.")
        print("Make sure dataset folder project ke andar hai.")
        return

    # Create destination emotion folders
    for emotion in set(EMOTIONS.values()):
        os.makedirs(
            os.path.join(DEST_FOLDER, emotion),
            exist_ok=True
        )

    total_files = 0

    # Actor_01 to Actor_24
    for actor_number in range(1, 25):

        actor_folder = f"Actor_{actor_number:02d}"

        actor_path = os.path.join(
            SOURCE_FOLDER,
            actor_folder
        )

        if not os.path.isdir(actor_path):
            print(f"⚠️ {actor_folder} nahi mila.")
            continue

        print(f"Processing {actor_folder}...")

        # Read audio files
        for filename in os.listdir(actor_path):

            if not filename.lower().endswith(".wav"):
                continue

            # Example:
            # 03-01-05-01-02-01-01.wav
            parts = filename.split("-")

            if len(parts) < 3:
                print(f"⚠️ Skipped: {filename}")
                continue

            # Emotion code is the 3rd part
            emotion_code = parts[2]

            if emotion_code not in EMOTIONS:
                print(f"⚠️ Unknown emotion: {filename}")
                continue

            emotion_name = EMOTIONS[emotion_code]

            # Source file
            source_file = os.path.join(
                actor_path,
                filename
            )

            # Destination file
            destination_file = os.path.join(
                DEST_FOLDER,
                emotion_name,
                filename
            )

            # Copy audio file
            shutil.copy2(
                source_file,
                destination_file
            )

            total_files += 1

    # Final result
    print()
    print("==========================================")
    print("✅ DATASET ORGANIZED SUCCESSFULLY")
    print("==========================================")
    print(f"Total audio files copied: {total_files}")

    print("\nEmotion folders:")

    for emotion in sorted(set(EMOTIONS.values())):

        folder = os.path.join(
            DEST_FOLDER,
            emotion
        )

        if os.path.exists(folder):

            count = len([
                f for f in os.listdir(folder)
                if f.lower().endswith(".wav")
            ])

            print(f"{emotion}: {count} files")


if __name__ == "__main__":
    organize_dataset()