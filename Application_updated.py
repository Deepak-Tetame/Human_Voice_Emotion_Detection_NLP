import numpy as np
import tensorflow as tf
import librosa
import pyaudio
import matplotlib.pyplot as plt
from tkinter import *
from PIL import Image, ImageTk
import pygame

# Load the trained model
model = tf.keras.models.load_model("emotion_detection_model_updated.h5")


# Function to preprocess live audio input
def preprocess_live_audio(audio):
    mel_spec = librosa.feature.melspectrogram(audio, sr=44100)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec = librosa.util.fix_length(mel_spec, 128, axis=1)
    return mel_spec


# Function to predict emotion from live input
def predict_emotion_live(audio):
    processed_audio = preprocess_live_audio(audio)
    prediction = model.predict(np.expand_dims(processed_audio, axis=0))
    emotion_label = np.argmax(prediction)
    return emotion_label


# Function to capture live audio from microphone
def capture_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 5

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []

    print("Recording...")

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio_data = np.frombuffer(b''.join(frames), dtype=np.float32)

    return audio_data


def play_song(emotion_label):
    songs = {
        0: 'MyWay.mp3',
        1: 'IWantItThatWay.mp3',
        2: 'Zach_Seabaugh_-_Christmas_Lights.mp3',
        3: 'See-You-Again(musicdownload.cc).mp3',
        4: 'Kuchh To Log Kahenge.mp3',
        5: 'Shivba-Raja_320(PagalWorld).mp3',
        6: 'Bhaag D.k. Bose, Aandhi Aayi Delhi Belly 320 Kbps.mp3',
        7: 'Rani-Mazya-Malyamandi.mp3'
    }
    pygame.mixer.init()
    song_path = songs.get(emotion_label)
    pygame.mixer.music.load(song_path)
    pygame.mixer.music.play()


def start_recording():
    audio_data = capture_audio()
    emotion_label = predict_emotion_live(audio_data)
    result_label.config(text="Predicted Emotion: " + emotions[emotion_label])
    # Plot the spectrogram
    plt.figure(figsize=(10, 4))

    # Plot waveform
    plt.subplot(2, 1, 1)
    plt.plot(np.linspace(0, len(audio_data) / 44100, len(audio_data)), audio_data)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Plot spectrogram
    plt.subplot(2, 1, 2)
    plt.imshow(preprocess_live_audio(audio_data).squeeze(), aspect='auto', origin='lower', cmap='inferno')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')

    # Save the spectrogram as a JPEG image
    output_file = "spectrogram.jpg"
    plt.savefig(output_file, format='jpg')

    # Display the spectrogram image
    image_path = output_file
    image = Image.open(image_path)
    image = image.resize((600, 300))

    # Convert the image to a format that Tkinter can handle
    photo = ImageTk.PhotoImage(image)

    # Create a label widget to display the image
    canvas.delete("all")
    canvas.create_image(0, 0, anchor=NW, image=photo)
    canvas.photo = photo
    play_song(emotion_label)


# Create Tkinter window
root = Tk()
root.geometry("1000x600")
root.title("Emotion Detection")
root.configure(bg="white")

# Add background image
background_image = Image.open("ai-technology-brain-background-digital-transformation-concept.jpg")
width, height = root.winfo_width(), root.winfo_height()
background_image = background_image.resize((1200, 800))
background_photo = ImageTk.PhotoImage(background_image)
background_label = Label(root, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

label1 = Label(root, text="Human Voice Emotion Detection", bg="cyan", font="Helvetica 24 bold")
label1.pack(fill=X, pady=15)

# Create a frame for buttons
button_frame = Frame(root, bg="cyan", relief=SUNKEN)
button_frame.pack(fill=X, pady=20, padx=10)

# Create a button to start recording
record_button = Button(button_frame, text="Start Recording", font="Helvetica 18 bold", command=start_recording,
                       borderwidth=10, fg="black",
                       relief=RAISED)
record_button.pack(side=LEFT, padx=20)

# Create a label to display the predicted emotion
result_label = Label(root, text="Predicted Emotion: ", font="Helvetica 18 bold", bg="white")
result_label.pack(side=TOP, padx=20, pady=10, anchor=W)

# Create a canvas to display the spectrogram image
canvas = Canvas(root, width=600, height=300, bg="gray")
canvas.pack(pady=10)

emotions = {
    0: 'Neutral',
    1: 'Calm',
    2: 'Happy',
    3: 'Sad',
    4: 'Angry',
    5: 'Fearful',
    6: 'Disgust',
    7: 'Surprised'
}

root.mainloop()
