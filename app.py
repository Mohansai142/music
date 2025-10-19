from flask import Flask, render_template, Response, jsonify, redirect, url_for
from camera import VideoCamera, get_final_emotion_index, music_rec, emotion_dict
import pandas as pd
import threading

app = Flask(__name__)

# Global state control for the two-step process
is_detecting = False 
current_camera = None

# Global variables for data passing
last_emotion_text = "Neutral"
headings = ("Name","Album","Artist")
df1 = pd.DataFrame(columns=headings) 

# --- NEW ROUTES FOR PAGE SEPARATION ---

@app.route('/')
def home():
    """Renders the simple landing page with navigation links (create home.html)."""
    global is_detecting
    # Ensure detection is stopped/reset when navigating back to home
    is_detecting = False 
    
    return render_template('home.html')

@app.route('/detector')
def detector():
    """Renders the page where the user can start/stop emotion detection (create detector.html)."""
    return render_template('detector.html')

@app.route('/menu')
def menu():
    """NEW ROUTE: Renders the content selection menu page."""
    global last_emotion_text
    # Pass the stored emotion to the menu template
    return render_template('menu.html', emotion=last_emotion_text)

@app.route('/recommendations/<content_type>')
def recommendations(content_type):
    """UPDATED ROUTE: Renders the page to display the chosen content recommendations."""
    global df1
    global last_emotion_text
    
    data = None
    emotion = last_emotion_text
    
    if content_type == 'songs':
        # Logic for Songs (using pre-computed df1)
        data = df1
        display_headings = ("Name", "Album", "Artist")
        page_title = f"Song Recommendations for Your {emotion} Mood"
        
    elif content_type == 'videos':
        # Placeholder logic for Videos
        data_list = [
            {"Name": f"Calm Nature Walk for {emotion}", "URL": "https://youtu.be/somevideo1"},
            {"Name": f"Quick Workout Motivation for {emotion}", "URL": "https://youtu.be/somevideo2"},
            {"Name": f"Funny Cat Compilation for {emotion}", "URL": "https://youtu.be/somevideo3"},
        ]
        data = pd.DataFrame(data_list)
        display_headings = ("Video Title", "Link")
        page_title = f"Video Suggestions for Your {emotion} Mood"

    elif content_type == 'mudras':
        # Placeholder logic for Acupressure Mudras
        data_list = [
            {"Name": "Gyan Mudra", "Benefit": "Improves concentration and memory, stimulates the root chakra."},
            {"Name": "Prana Mudra", "Benefit": "Enhances vitality and reduces fatigue, good for eyes."},
            {"Name": "Apan Vayu Mudra", "Benefit": "Heart health, regulates blood pressure and digestion."},
        ]
        data = pd.DataFrame(data_list)
        display_headings = ("Mudra Name", "Benefit")
        page_title = f"Acupressure Mudras for {emotion}"
        
    elif content_type == 'quotes':
        # Placeholder logic for Motivational Quotes
        data_list = [
            {"Quote": "The best way to predict the future is to create it.", "Source": "Peter Drucker"},
            {"Quote": "Believe you can and you're halfway there.", "Source": "Theodore Roosevelt"},
            {"Quote": "Strive not to be a success, but rather to be of value.", "Source": "Albert Einstein"},
        ]
        data = pd.DataFrame(data_list)
        display_headings = ("Quote", "Source")
        page_title = f"Motivational Quotes for {emotion}"
        
    else:
        # Default or error case
        display_headings = ("Content",)
        data = pd.DataFrame([{"Content": "Invalid content type selected."}])
        page_title = "Content Not Found"
    
    
    # Now, render the template with the dynamic data and headings
    return render_template('recommendations.html', 
                           headings=display_headings, 
                           data=data, 
                           emotion=emotion,
                           page_title=page_title)
    
# --- EXISTING LOGIC ROUTES (KEPT FOR FUNCTIONALITY) ---

def gen(camera):
    """Generator function that yields video frames until stopped."""
    global is_detecting
    while is_detecting:
        # Note: The camera.get_frame() function is presumed to handle the 
        # internal tracking of emotion to allow get_final_emotion_index() to work.
        frame, _ = camera.get_frame() 
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
    # Stop the camera when the video stream thread ends (after stop_detection is called)
    if camera:
        camera.stop()

@app.route('/video_feed')
def video_feed():
    """Route to start and serve the real-time video stream."""
    global is_detecting
    global current_camera
    
    if not is_detecting:
        # Initialize and flag detection as active
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
    global last_emotion_text # Now using the global emotion variable
    
    # 1. Stop the stream.
    is_detecting = False 
    
    # 2. Get the final data and store the emotion
    final_emotion_index = get_final_emotion_index()
    last_emotion_text = emotion_dict.get(final_emotion_index, "Neutral")
    
    # 3. Generate the song recommendations only (as this takes time)
    # The non-song content will be generated on the fly in the recommendations route.
    df1 = music_rec(final_emotion_index) 
    df1 = df1.head(15) 
    
    # 4. Cleanup
    current_camera = None

    # 5. Redirect the user to the MENU page after processing
    return jsonify({
        'emotion': last_emotion_text,
        'songs_available': True,
        'redirect_url': url_for('menu') # CRUCIAL CHANGE: Redirects to the menu
    })

@app.route('/t')
def gen_table():
    """API endpoint to return the current song data as JSON (for debugging/testing)."""
    global df1
    return df1.to_json(orient='records')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)