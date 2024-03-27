import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


# Function to load audio files and extract features
def load_audio_files_and_labels(data_dir):
    audio_files = []
    labels = []
    emotions = {'01': 0, '02': 1, '03': 2, '04': 3, '05': 4, '06': 5, '07': 6,
                '08': 7}  # Map file codes to emotion labels

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                label = emotions[file.split('-')[2]]
                audio_files.append(file_path)
                labels.append(label)

    return audio_files, labels


# Preprocess the audio files (convert to spectrograms)
def preprocess_audio(audio_files, labels):
    X = []
    y = []
    for file, label in zip(audio_files, labels):
        # Load audio file
        audio, sr = librosa.load(file, sr=None)
        # Extract features (e.g., mel spectrogram)
        mel_spec = librosa.feature.melspectrogram(audio, sr=sr)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to decibels
        # Resize to a fixed shape (e.g., 128x128)
        mel_spec = librosa.util.fix_length(mel_spec, 128, axis=1)
        # Add channel dimension
        mel_spec = np.expand_dims(mel_spec, axis=-1)
        X.append(mel_spec)
        y.append(label)
    return np.array(X), np.array(y)


# Load audio files and labels
data_dir = 'speech-emotion-recognition-ravdess-data'
audio_files, labels = load_audio_files_and_labels(data_dir)

# Preprocess the audio files
X, y = preprocess_audio(audio_files, labels)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(8, activation='softmax')  # 8 classes for 8 emotions
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train,
                    epochs=10,
                    validation_data=(X_test, y_test),
                    batch_size=32,
                    verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print("Test Accuracy:", accuracy)

# Save the model
model.save("emotion_detection_model_updated.h5")
