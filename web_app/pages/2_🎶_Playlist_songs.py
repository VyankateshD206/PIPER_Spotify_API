import streamlit as st
import requests
from datetime import datetime

API_BASE_URL = 'https://api.spotify.com/v1/'
st.set_page_config(page_title="Playlist songs", page_icon="🎶" )
st.title("Songs based on playlist")

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

if 'access_token' in st.session_state:
    def get_current_user_info():
        headers = {
            'Authorization': f"Bearer {st.session_state['access_token']}"
        }
        response = requests.get(API_BASE_URL + 'me', headers=headers)
        user_info = response.json()
        return user_info

    def get_artist_details(artist_id):
        headers = {
            'Authorization': f"Bearer {st.session_state['access_token']}"
        }
        response = requests.get(API_BASE_URL + f'artists/{artist_id}', headers=headers)
        artist_details = response.json()

        return artist_details

    def get_current_user_followed_artist():
        headers = {
            'Authorization': f"Bearer {st.session_state['access_token']}"
        }
        # Get followed artists
        followed_artists_response = requests.get(API_BASE_URL + 'me/following?type=artist', headers=headers)
        followed_artists = followed_artists_response.json()   
        return followed_artists

    def get_playlist_tracks(playlist_id):
        if 'access_token' not in st.session_state:
            st.warning("Please log in to Spotify.")
            return

        if datetime.now().timestamp() > st.session_state.expire_at:
            st.warning("Token expired. Please log in again.")
            return

        headers = {
            'Authorization': f"Bearer {st.session_state['access_token']}"
        }
        response = requests.get(API_BASE_URL + f'playlists/{playlist_id}/tracks', headers=headers)
        playlist_tracks = response.json()
        
        return playlist_tracks    

    def get_playlists():
        headers = {
            'Authorization': f"Bearer {st.session_state['access_token']}"
        }
        response = requests.get(API_BASE_URL + 'me/playlists', headers=headers)
        playlists = response.json()
        return playlists
        #use:# for playlist in playlists['items']:
            # st.write(f"- {playlist['name']} (ID: {playlist['id']})")

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

    #select playlist by user 
    def select_playlist(limit, access_token):
        limit1 = int(limit.split('-')[0])
        playlists = get_playlists()
        k=0
        for playlist in playlists['items']:       
            playlist_id = playlist['id']
            k=k+1
            if k==limit1:
                break
        if 'playlist_id' not in st.session_state:
            st.session_state.playlist_id = playlist_id

        songs= get_playlist_tracks(playlist_id)
        st.write(playlist['name'])
        st.write("=======================your songs fetch for selected playlist=============================")
        for song in songs['items']:
            song = song['track']
            st.write(f"{song['name']} {song['artists'][0]['name']} {song['artists'][0]['id']}")
            preview_url = song['preview_url'] if 'preview_url' in song else None
            if st.button(f"Remove {song['name']}", key=song['name']):
                remove_track_from_playlist(song['id'], playlist_id, access_token)

            # Display the audio player if a preview URL is available
            if preview_url:
                    st.audio(preview_url)
            else:
                    st.write("Preview not available")
            

    #to get songs of a selected playlist:
    access_token = st.session_state.access_token
    playlists = get_playlists()
    playlist_options =[]
    i=1
    if 'items' in playlists:
        st.write("playlist is there")
        for playlist in playlists['items']:  
            playlist_options.append(f"{i}-{playlist['name']}")
            i=i+1
    st.subheader("to get songs of a selected playlist:")
    selectP = st.selectbox('Select playlist of which you want songs:', playlist_options)
    if 'selected_playlist' not in st.session_state:
            st.session_state.selected_playlist = selectP
    #st.session_state.selected_playlist = selectP
    if st.session_state.selected_playlist is not None:
        if st.button("Get songs of Playlist:"):
            st.success('Button clicked!') 
            select_playlist(selectP, access_token)

else:
    # error message if the 'access_token' key is not found
    st.error("Please go to the HOME page and log in")