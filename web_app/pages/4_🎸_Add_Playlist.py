import streamlit as st
import requests

API_BASE_URL = 'https://api.spotify.com/v1/'
st.set_page_config(page_title="New Playlist", page_icon="🎸" )
st.title("Create New Playlist")
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
        if response.status_code == 201:
            st.success("Track added to playlist successfully!")
        else:
            st.success("Track added to playlist successfully!")

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

    access_token = st.session_state.access_token
    st.write("Enter a name for your playlist:")
    playlist_name = st.text_input("Enter a name for the new playlist:")
    if playlist_name != "":
        if 'playlist_id' not in st.session_state:
            playlist_id= add_playlist(playlist_name)
            st.session_state.playlist_id = playlist_id

        display_top_songs(st.session_state.playlist_id, access_token)

else:
    st.error("Please go to the HOME page and log in")
