import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import joblib
import streamlit as st
import streamlit as st
import requests
st.set_page_config(page_title="Mood Based Playlist", page_icon="▶️" )

API_BASE_URL = 'https://api.spotify.com/v1/'
st.title("Mood Based Playlist of Your Top Tracks")
page_bg_img=""" 
<style>
[data-testid="stAppViewContainer"]{
background-image: url("https://t3.ftcdn.net/jpg/05/50/05/52/360_F_550055239_zK6qJTCOfodrftSLJM7bjcoUnF6lIl6Y.jpg");
background-size: cover;
animation: animateBackground 10s infinite linear;
}
[data-testid="stSidebarNavLink"]{
    color: #FFFFFF;
    font-size: 20px;
    font-weight: bold;
}
@keyframes animateBackground {
    0% {
        background-position: 0% 0%;
    }
    100% {
        background-position: 100% 100%;
    }
}
[data-testid="stHeader"]{
background : rgba(0,0,0,0);
}
[data-testid="stToolbar"]{
right: 2rem;
}
[data-testid="stSidebar"] > div:first-child{
background-image: url("https://th.bing.com/th/id/OIP.J0_5W76CuWJweiL8wPwScQHaD7?w=337&h=180&c=7&r=0&o=5&dpr=1.3&pid=1.7");
background-position: centre;
}

    
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Define the same neural network model class
class MoodClassifier(nn.Module):
    def __init__(self):
        super(MoodClassifier, self).__init__()
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 5)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x



if 'access_token' in st.session_state:
    access_token = st.session_state.access_token

    # Function to fetch user's top tracks
    def get_top_tracks(access_token):
        headers = {
            'Authorization': f"Bearer {access_token}"
        }
        params = {
            'limit': 50,
            'time_range': 'short_term'
        }
        response = requests.get('https://api.spotify.com/v1/me/top/tracks', headers=headers, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            return None

    # Function to fetch audio features for a track
    def get_audio_features(track_id, access_token):
        headers = {
            'Authorization': f"Bearer {access_token}"
        }
        response = requests.get(f'https://api.spotify.com/v1/audio-features/{track_id}', headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            return None

    # Function to extract relevant features from top tracks
    def extract_track_features(top_tracks, access_token):
        track_data = []
        for track in top_tracks['items']:
            audio_features = get_audio_features(track['id'], access_token)
            if audio_features:
                track_info = {
                    'track_id': track['id'],
                    'user_id': track['name'],
                    'danceability': audio_features['danceability'],
                    'valence': audio_features['valence'],
                    'energy': audio_features['energy'],              
                    'tempo': audio_features['tempo'],
                    'loudness': audio_features['loudness'],
                }
                track_data.append(track_info)
        return track_data


    def add_playlist(name):
        headers = {
            'Authorization': f"Bearer {st.session_state.access_token}",
            'Content-Type': 'application/json',
        }
        data = {
            'name': name,
            'public': False  # You can set it to True if you want the playlist to be public
        }
        response = requests.post(API_BASE_URL + 'me/playlists', headers=headers, json=data)
        
        if response.status_code == 201:
            playlist_data = response.json()
            playlist_id = playlist_data['id']
            st.success("Playlist created successfully!")
            return playlist_id
        else:
            st.write("Failed to create playlist. Error:", response.text)

    def get_top_songs(access_token):
        headers = {
            'Authorization': f"Bearer {access_token}"
        }
        response = requests.get(API_BASE_URL + 'me/top/tracks?limit=20', headers=headers)
        if response.status_code == 200:
            return response.json()['items']
        else:
            st.write("Failed to fetch top songs. Error:", response.text)
            return []

    def add_track_to_playlist(track_id, playlist_id, access_token):
        headers = {
            'Authorization': f"Bearer {access_token}",
            'Content-Type': 'application/json',
        }
        data = {
            'uris': ['spotify:track:' + track_id]
        }
        response = requests.post(API_BASE_URL + f'playlists/{playlist_id}/tracks', headers=headers, json=data)
        # if response.status_code == 201:
        #     st.success("Track added to playlist successfully!")
        # else:
        #     st.success("Track added to playlist successfully!")

    def remove_track_from_playlist(track_id, playlist_id, access_token):
        headers = {
            'Authorization': f"Bearer {access_token}",
            'Content-Type': 'application/json',
        }
        # Construct the request body with the track URI
        data = {
            'tracks': [
                {
                    'uri': f"spotify:track:{track_id}"
                }
            ]
        }
        # Send a DELETE request to remove the track from the playlist
        response = requests.delete(API_BASE_URL + f'playlists/{playlist_id}/tracks', headers=headers, json=data)
        if response.status_code == 200:
            st.success("Track removed from playlist successfully!")
            print("Done")
        else:
            st.error("Failed to remove track from playlist. Error:", response.text)
            print(response.text)

    def display_top_songs(playlist_id, access_token):
        top_songs = get_top_songs(access_token)
        for index, song in enumerate(top_songs, start=1):
            song_name = song['name']
            song_id = song['id']
            artist_names = ', '.join([artist['name'] for artist in song['artists']])
            display_text = f"{index}. {song_name} by {artist_names}"
            st.write(display_text)
            if st.button("Add to Playlist", key= song_id):
                add_track_to_playlist(song_id, playlist_id, access_token)
            if st.button(f"Remove {song_name}", key=index):
                remove_track_from_playlist(song_id, playlist_id, access_token)


    # Fetch top tracks
    top_tracks = get_top_tracks(access_token)
    
    if top_tracks:
        # Extract features
        if 'track_data' not in st.session_state:     
            st.session_state.track_data = extract_track_features(top_tracks, access_token)
            st.write("Please wait till the data is fetched.......")
            # Display DataFrame
            df = pd.DataFrame(st.session_state.track_data)       
            
            # Save DataFrame to CSV
            df.to_csv('top_tracks_features.csv', index=False)
            st.success("DataFrame saved to 'top_tracks_features.csv'")
        df = pd.DataFrame(st.session_state.track_data)
        st.write('Top Tracks:')
        st.dataframe(df)
    else:
        st.error("Failed to fetch top tracks. Please check your authentication.")
    
    # Load the saved model and scaler
    model = MoodClassifier()
    model.load_state_dict(torch.load('web_app/mood_predictor_model1.pth'))
    model.eval()
    scaler = joblib.load('web_app/scaler1.joblib')

    # Load new data
    new_data = pd.read_csv('top_tracks_features.csv')

    # Normalize the new song data
    new_features = new_data[['danceability', 'energy', 'valence', 'tempo', 'loudness']]
    new_features_scaled = scaler.transform(new_features)

    # Convert to PyTorch tensor
    new_features_tensor = torch.tensor(new_features_scaled, dtype=torch.float32)

    # Predict the mood
    with torch.no_grad():
        predictions = model(new_features_tensor)
        _, predicted_moods = torch.max(predictions, 1)

    new_data['predicted_mood'] = predicted_moods.numpy()
    st.write({
    'Happy': 0,
    'Calm': 1,
    'Neutral': 2,
    'Sad': 3,
    'Very Sad': 4
    })
    
    the_mood = st.number_input("enter the number for your MOOD")
    st.session_state.the_mood = the_mood
    st.write("the mood is ", {st.session_state.the_mood})
    # Suggest five songs that are predicted 
    required_songs = new_data[new_data['predicted_mood'] == st.session_state.the_mood]
    # if required_songs['track_id']=="":
    #     if the_mood==0:
    #         the_mood= 1
    #     if the_mood==4:
    #         the_mood=3     
    #     required_songs = new_data[new_data['predicted_mood'] == the_mood]
    # Select the first five happy songs
    five_songs = required_songs.head(5)

    
    playlist_name = st.text_input("Enter a name for the new playlist:")
    if playlist_name != "":       
        if 'playlist_id' not in st.session_state: 
            playlist_id= add_playlist(playlist_name)       
            st.session_state.playlist_id = playlist_id
        url_playlist = f"https://open.spotify.com/playlist/{st.session_state.playlist_id}"

        # display_top_songs(st.session_state.playlist_id, access_token)
        for i in five_songs['track_id'] :
            add_track_to_playlist(i, st.session_state.playlist_id, access_token)
        st.success("Tracks added to playlist successfully!")
        st.header(f"Go to [Spotify Web Player]({url_playlist})")
        st.warning("Once playlist is created You need to LOG in Again to make new one")
else:
    st.error("Please go to the HOME page and log in")
