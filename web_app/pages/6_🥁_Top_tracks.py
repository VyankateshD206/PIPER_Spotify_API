import streamlit as st
import pandas as pd
import requests

# Spotify API credentials
CLIENT_ID = "<your client id>"
CLIENT_SECRET = "<your client secret>"
REDIRECT_URI = "<your redirect URI>"  # e.g., 'http://localhost:8888/callback'
SCOPE = "user-top-read"

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
if 'access_token' in st.session_state:
# Create Streamlit app
    def main():
        st.title('My Top Spotify Tracks')
        access_token = st.session_state.access_token
        
        if access_token:
            # Fetch top tracks
            top_tracks = get_top_tracks(access_token)
            
            if top_tracks:
                # Extract features
                track_data = extract_track_features(top_tracks, access_token)
                
                # Display DataFrame
                df = pd.DataFrame(track_data)
                st.write('Top Tracks:')
                st.dataframe(df)
                
                # Save DataFrame to CSV
                if st.button('Save as CSV'):
                    df.to_csv('top_tracks_features.csv', index=False)
                    st.success("DataFrame saved to 'top_tracks_features.csv'")
            else:
                st.error("Failed to fetch top tracks. Please check your authentication.")
        else:
            st.error("Access token is missing. Please authenticate with Spotify.")
else:
    st.error("Access token is missing. Please authenticate with Spotify.")        

if __name__ == '__main__':
    main()
