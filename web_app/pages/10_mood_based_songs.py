import streamlit as st
import requests
import pandas as pd
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
        return response.json()['items']
    else:
        return None

# Function to search for tracks based on mood
def search_tracks_by_mood(mood, access_token):
    url = 'https://api.spotify.com/v1/search'
    params = {
        'q': f'{mood}',
        'type': 'track',
        'market': 'US',
        'limit': 10
    }
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200:
        return response.json()['tracks']['items']
    else:
        return None

# Function to fetch audio features for a track
def get_audio_features(track_id, access_token):
    url = f'https://api.spotify.com/v1/audio-features/{track_id}'
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Function to filter tracks by valence
def filter_tracks_by_valence(tracks, threshold1, threhold2):
    filtered_tracks = []
    count=0
    for track in tracks:
        if 'valence' in track and threhold2 >=track['valence'] >= threshold1:                   
            filtered_tracks.append(track)
            count += 1
            if count == 10:
                break
    return filtered_tracks

# Streamlit app
def main():
    st.title('Search Tracks from Top Tracks')
    
    # Authenticate with Spotify and get access token
    access_token = st.session_state.access_token
    mood = st.text_input('Enter the mood:')
    if(mood=='happy'):
        threshold1 = 0.6
        threshold2= 0.98
    elif(mood=='sad'):
        threshold1 = 0.1
        threshold2 = 0.5
    else:
        threshold1= 0.4
        threshold2 = 0.7
    
    if access_token:
        # Fetch user's top tracks
        top_tracks = get_top_tracks(access_token)
        
        if top_tracks:
            # Fetch audio features for each top track
            top_tracks_with_features = []
            for track in top_tracks:
                track_id = track['id']
                audio_features = get_audio_features(track_id, access_token)
                if audio_features:
                    track['valence'] = audio_features['valence']
                    top_tracks_with_features.append(track)
            
            # Filter top tracks by valence (happy mood)
            happy_tracks = filter_tracks_by_valence(top_tracks_with_features, threshold1, threshold2)
            # Fetch additional happy tracks if needed
            if len(happy_tracks) < 10:
                additional_tracks = search_tracks_by_mood(mood, access_token)
                if additional_tracks:
                    happy_tracks += additional_tracks[:10-len(happy_tracks)]
            if happy_tracks:
                df = pd.DataFrame(happy_tracks)
                st.write(f"Found {len(happy_tracks)} {mood} tracks from your top tracks:")
                st.dataframe(df)
                if st.button('Save CSV'):
                    df.to_csv(f'{mood}_tracks.csv', index=False)
                    st.success(f"DataFrame saved to '{mood}_tracks.csv'")
                #for track in happy_tracks:
                    #st.write(f"Name: {track['name']}, Artist: {track['artists'][0]['name']}")
            else:
                st.write(f"No {mood} tracks found in your top tracks.")
        else:
            st.write("Failed to fetch your top tracks.")
    else:
        st.write("Failed to authenticate with Spotify.")

if __name__ == "__main__":
    main()
