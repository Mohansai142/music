import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing import image
import pandas as pd
from threading import Thread
import os # Added for better error handling

# --- Model and Classifier Loading ---
# Ensure 'haarcascade_frontalface_default.xml' is in the same directory
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 
ds_factor=0.6

# Build and load the emotion model
emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
# Ensure 'model.h5' is in the same directory
emotion_model.load_weights('model.h5') 

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0:"Angry",1:"Disgusted",2:"Fearful",3:"Happy",4:"Neutral",5:"Sad",6:"Surprised"}

# File paths for music CSVs
music_dist={
    0:"songs/angry.csv".strip(), 1:"songs/disgusted.csv".strip(), 
    2:"songs/fearful.csv".strip(), 3:"songs/happy.csv".strip(), 
    4:"songs/neutral.csv".strip(), 5:"songs/sad.csv".strip(), 
    6:"songs/surprised.csv".strip()
}

# --- GLOBAL STATE DECLARATION ---
global current_emotion_index 
current_emotion_index = 4 # Default to Neutral
global last_frame1 
last_frame1 = np.zeros((500, 600, 3), dtype=np.uint8) 

# --- Helper Functions ---

def get_final_emotion_index():
    global current_emotion_index
    return current_emotion_index

# CRITICAL FIX: Added default argument (emotion_index=4) to prevent TypeError 
# when music_rec() is called without arguments in app.py startup.
def music_rec(emotion_index=4): 
    """Retrieves the song recommendation based on the provided emotion index, including URL."""
    if emotion_index not in music_dist:
        return pd.DataFrame(columns=['Name','Album','Artist', 'URL']) 

    music_file_path = music_dist.get(emotion_index)
    if not music_file_path or not os.path.exists(music_file_path):
        # Improved error handling for missing CSV
        print(f"ERROR: Music file not found for emotion index {emotion_index} at path {music_file_path}. Please run Spotipy.py.")
        return pd.DataFrame(columns=['Name','Album','Artist', 'URL']) 
        
    try:
        df = pd.read_csv(music_file_path)
    except Exception as e:
        print(f"Error reading CSV {music_file_path}: {e}")
        return pd.DataFrame(columns=['Name','Album','Artist', 'URL'])

    # Ensure the required 'URL' column is present
    required_cols = ['Name','Album','Artist', 'URL']
    if all(col in df.columns for col in required_cols):
        df = df[required_cols]
    else:
        # Fallback if the CSV is missing the URL (meaning Spotipy.py wasn't run)
        print("Warning: CSV file is missing the 'URL' column. Please run the updated Spotipy.py script to regenerate all CSVs.")
        # Attempt to recover with existing columns and an empty URL column
        df = df[['Name','Album','Artist']]
        df['URL'] = '' 

    df = df.head(15)
    return df

# --- WebcamVideoStream Class (Multi-threading for camera) ---
class WebcamVideoStream:
         
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        # Set buffer size to 1 to ensure we get the latest frame
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self
         
    def update(self):
        while True:
            if self.stopped:
                self.stream.release() 
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame
         
    def stop(self):
        self.stopped = True

# --- VideoCamera Class (Core Logic) ---
class VideoCamera(object):
    
    def __init__(self):
        self.vs = WebcamVideoStream(src=0).start()
        
    def stop(self):
        self.vs.stop()

    def get_frame(self):
        global current_emotion_index
        global last_frame1
        
        image = self.vs.read()
        
        # Handle case where camera read fails or image is empty
        if image is None or image.size == 0:
            ret, jpeg = cv2.imencode('.jpg', last_frame1)
            return jpeg.tobytes(), current_emotion_index
            
        # Image processing
        if image.shape[0] > 0 and image.shape[1] > 0:
            image=cv2.resize(image,(600,500))
        
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        face_rects=face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in face_rects:
            cv2.rectangle(image,(x,y-50),(x+w,y+h+10),(0,255,0),2)
            roi_gray_frame = gray[y:y + h, x:x + w]
            
            # Check for valid ROI before resize/prediction
            if roi_gray_frame.shape[0] == 0 or roi_gray_frame.shape[1] == 0:
                 continue 
                 
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            
            # Predict emotion
            prediction = emotion_model.predict(cropped_img, verbose=0) 
            maxindex = int(np.argmax(prediction))
            
            current_emotion_index = maxindex 
            
            cv2.putText(image, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            break
            
        last_frame1 = image.copy()
        
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes(), current_emotion_index