import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import pandas as pd
from threading import Thread
import os

# --- Model and Classifier Loading ---
# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
ds_factor = 0.6

# --- Emotion Model Definition ---
emotion_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# Load model weights
emotion_model.load_weights('model.h5')

cv2.ocl.setUseOpenCL(False)

emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful", 
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
}

# File paths for emotion-wise song lists
music_dist = {
    0: "songs/angry.csv",
    1: "songs/disgusted.csv",
    2: "songs/fearful.csv",
    3: "songs/happy.csv",
    4: "songs/neutral.csv",
    5: "songs/sad.csv",
    6: "songs/surprised.csv"
}

# --- Global Variables ---
current_emotion_index = 4  # Neutral
last_frame1 = np.zeros((500, 600, 3), dtype=np.uint8)

# --- Helper Functions ---
def get_final_emotion_index():
    global current_emotion_index
    return current_emotion_index

def music_rec(emotion_index=4):
    """Retrieve song recommendations for the detected emotion."""
    required_cols = ['Name', 'Album', 'Artist', 'URL']
    empty_df = pd.DataFrame(columns=required_cols)

    if emotion_index not in music_dist:
        return empty_df

    music_file_path = music_dist.get(emotion_index)
    if not os.path.exists(music_file_path):
        print(f"ERROR: Missing CSV for emotion index {emotion_index} => {music_file_path}")
        return empty_df

    try:
        df = pd.read_csv(music_file_path)
    except Exception as e:
        print(f"Error reading {music_file_path}: {e}")
        return empty_df

    if all(col in df.columns for col in required_cols):
        df = df[required_cols]
    else:
        print(f"Warning: {music_file_path} is missing 'URL'. Adding placeholder column.")
        current_cols = [col for col in ['Name', 'Album', 'Artist'] if col in df.columns]
        df = df[current_cols]
        df['URL'] = '#'

    return df.head(15)

# --- Video Stream Class ---
class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.stream.read()
        self.stream.release()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# --- Emotion Detection via Video Camera ---
class VideoCamera:
    def __init__(self):
        self.vs = WebcamVideoStream(src=0).start()

    def stop(self):
        self.vs.stop()

    def get_frame(self):
        global current_emotion_index, last_frame1

        frame = self.vs.read()
        if frame is None or frame.size == 0:
            ret, jpeg = cv2.imencode('.jpg', last_frame1)
            return jpeg.tobytes(), current_emotion_index

        frame = cv2.resize(frame, (600, 500))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]

            if roi_gray.size == 0:
                continue

            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = emotion_model.predict(cropped_img, verbose=0)
            maxindex = int(np.argmax(prediction))

            current_emotion_index = maxindex

            cv2.putText(frame, emotion_dict[maxindex], (x + 20, y - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            break

        last_frame1 = frame.copy()
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes(), current_emotion_index
