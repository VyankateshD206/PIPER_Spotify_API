import streamlit as st
import requests
from datetime import datetime

SPOTIPY_REDIRECT_URI = 'http://localhost:8501'
API_BASE_URL = 'https://api.spotify.com/v1/'

st.set_page_config(page_title="Artist", page_icon="🎙️" )
st.title("Followed Artists")

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

# To get songs based on artists     
st.subheader("To get songs based on artist:")
if 'access_token' in st.session_state:
    def get_playlists():
        
        headers = {
            'Authorization': f"Bearer {st.session_state['access_token']}"
        }
        response = requests.get(API_BASE_URL + 'me/playlists', headers=headers)
        playlists = response.json()
        return playlists



    def get_current_user_followed_artist():
        headers = {
            'Authorization': f"Bearer {st.session_state['access_token']}"
        }
        # Get followed artists
        followed_artists_response = requests.get(API_BASE_URL + 'me/following?type=artist', headers=headers)
        followed_artists = followed_artists_response.json()   
        return followed_artists

    def get_playlists():
        
        headers = {
            'Authorization': f"Bearer {st.session_state['access_token']}"
        }
        response = requests.get(API_BASE_URL + 'me/playlists', headers=headers)
        playlists = response.json()
        return playlists

    def get_playlist_tracks(playlist_id):
        if 'access_token' not in st.session_state:
            st.warning("Please log in to Spotify.")
            return

        headers = {
            'Authorization': f"Bearer {st.session_state['access_token']}"
        }
        response = requests.get(API_BASE_URL + f'playlists/{playlist_id}/tracks', headers=headers)
        playlist_tracks = response.json()
        
        return playlist_tracks

    #songs_by_artist :=>
    def songs_by_artist(selected_artist, limit):
        limit1 = int(limit.split('-')[0])
        playlists = get_playlists()
        k=0
        for playlist in playlists['items']:       
            playlist_id = playlist['id']
            k=k+1
            if k==limit1:
                break
        songs= get_playlist_tracks(playlist_id)

        artist_ID = str(selected_artist.split('-')[1])
        tracks=[]
        for song in songs['items']:    
            song = song['track']# for streamlit user will give name and we ll run loop to get its id
            
            if(artist_ID == song['artists'][0]['id']):
                tracks.append(f"{song['name']} by {song['artists'][0]['name']}")
                s=song['id']
                audio_features_response = requests.get(API_BASE_URL + f'audio-features/{s}', headers={'Authorization': f"Bearer {st.session_state['access_token']}"})
                audio_features = audio_features_response.json()

                # Get the preview URL for the track
                preview_url = song['preview_url'] if 'preview_url' in song else None

                # Display the audio player if a preview URL is available
                if preview_url:
                    st.audio(preview_url)
                    # Create a like button
                    
                else:
                    st.write("Preview not available")           
        return tracks

    #code to be executed:
    playlists = get_playlists()
    playlist_options =[]
    i=1
    if 'items' in playlists:
        for playlist in playlists['items']:  
            playlist_options.append(f"{i}-{playlist['name']}")
            i=i+1
        
    all_artists = get_current_user_followed_artist()
    artists_names=[]
    if 'artists' in all_artists:   
        for artist in all_artists['artists']['items']:
            artists_names.append(f"{artist['name']}-{artist['id']}")
    st.write(artists_names)
    selectP1 = st.selectbox('Select playlist for songs:', playlist_options)
    selectA = st.selectbox('Select artist:', artists_names)
    st.session_state.selected_playlist1 = selectP1
    st.session_state.selected_artist = selectA

    if st.button("Get Artist:"):
        tracks1=songs_by_artist(st.session_state.selected_artist, st.session_state.selected_playlist1 )
        st.success('Button clicked!')
        for track1 in tracks1:
            st.write(track1)

else:
    # Show an error message if the 'access_token' key is not found
    st.error("Please go to the HOME page and log in")
