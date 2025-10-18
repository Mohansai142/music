from flask import Flask, render_template, Response, jsonify
from camera import VideoCamera, get_final_emotion_index, music_rec, emotion_dict
import pandas as pd
import threading

app = Flask(__name__)

# Global state control for the two-step process
is_detecting = False 
current_camera = None

headings = ("Name","Album","Artist")
df1 = pd.DataFrame(columns=headings) 

@app.route('/')
def index():
    global is_detecting
    is_detecting = False 
    
    return render_template('index.html', headings=headings, data=df1, initial_emotion="Click 'Start Detection'")

def gen(camera):
    """Generator function that yields video frames until stopped."""
    global is_detecting
    while is_detecting:
        frame, _ = camera.get_frame() 
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    camera.stop()

@app.route('/video_feed')
def video_feed():
    """Route to start and serve the real-time video stream."""
    global is_detecting
    global current_camera
    
    if not is_detecting:
        current_camera = VideoCamera()
        is_detecting = True
        
    return Response(gen(current_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_detection')
def stop_detection():
    """Stops the video, gets the final emotion, and generates song recommendations."""
    global is_detecting
    global df1
    global current_camera
    
    # 1. Stop the stream
    is_detecting = False 
    
    # 2. Get the final data
    final_emotion_index = get_final_emotion_index()
    final_emotion_text = emotion_dict.get(final_emotion_index, "Neutral")
    
    df1 = music_rec(final_emotion_index) 
    
    # 3. Cleanup
    current_camera = None

    # 4. Return data as JSON (the song DataFrame now includes the URL)
    return jsonify({
        'emotion': final_emotion_text,
        'songs': df1.to_json(orient='records')
    })

@app.route('/t')
def gen_table():
    # Only returns the current (or last computed) song data
    return df1.to_json(orient='records')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)