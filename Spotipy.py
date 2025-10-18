import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time
import os

# NOTE: Your updated keys are here
auth_manager = SpotifyClientCredentials('61843d94b4b249088bcb21045206d1c9', '5998a6d4ab6044eca72083c4788f820f') 
sp = spotipy.Spotify(auth_manager=auth_manager)

def getTrackIDs(playlist_id): # NOTE: Only takes ONE argument now
    """Fetches track IDs from a Spotify playlist."""
    track_ids = []
    # Use sp.playlist(id) to handle the ID directly
    playlist = sp.playlist(playlist_id)
    
    # Check if 'tracks' and 'items' exist before iterating
    if 'tracks' in playlist and 'items' in playlist['tracks']:
        for item in playlist['tracks']['items']:
            track = item.get('track')
            if track and track.get('id'):
                track_ids.append(track['id'])
                
    return track_ids

def getTrackFeatures(id):
    """Fetches track features including the Spotify URL."""
    track_info = sp.track(id)

    name = track_info['name']
    album = track_info['album']['name']
    artist = track_info['album']['artists'][0]['name']
    
    # CRITICAL: Get the Spotify external URL for the track
    song_url = track_info['external_urls']['spotify'] 

    # Include the new URL in the returned data list
    track_data = [name, album, artist, song_url] 
    return track_data

# Playlist IDs
music_dist={0:"3yT3sMeDhNHloaBjvHLLtk", 1:"5AjguFfQcwckxklPO4XvT5", 2:"5q8sllODnbXCWJyha3cL5n",    3:"4jW37umAGFKr2oQRAk5pAe",4:"0u4VpvCZUw4fJfbTYdL6XP",5: "4jW37umAGFKr2oQRAk5pAe",6:"3yT3sMeDhNHloaBjvHLLtk"}

emotions = {
0: 'angry', 1: 'disgusted', 2: 'fearful',  
      3: 'surprised' ,4: "sad",5:"happy",6:"neutral"
    
}

# --- NEW/UNCOMMENTED DATA GENERATION LOOP ---
if __name__ == '__main__':
    if not os.path.exists('songs'):
        os.makedirs('songs')

    print("--- Starting Spotify Data Regeneration (with URLs) ---")
    
    for index, emotion_name in emotions.items():
        playlist_id = music_dist[index]
        print(f"Fetching songs for: {emotion_name}")
        
        # CRITICAL: Call the updated function with one argument
        track_ids = getTrackIDs(playlist_id) 
        
        track_list = []
        for track_id in track_ids:
            time.sleep(0.3)
            try:
                track_data = getTrackFeatures(track_id)
                track_list.append(track_data)
            except Exception as e:
                print(f"Error processing track {track_id}: {e}")
                continue
                
        # CRITICAL: The columns list now correctly includes 'URL'
        df = pd.DataFrame(track_list, columns=['Name', 'Album', 'Artist', 'URL']) 
        df.to_csv(f'songs/{emotion_name}.csv', index=False, encoding='utf-8')
        print(f"{emotion_name.capitalize()} CSV Generated with {len(df)} songs and URLs")
    
    print("--- Spotify Data Regeneration Complete ---")