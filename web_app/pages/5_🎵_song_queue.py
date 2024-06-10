import streamlit as st
import requests

st.set_page_config(page_title="Song Queue", page_icon="🎵" )
API_BASE_URL = 'https://api.spotify.com/v1/'

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
    def get_user_playlists(access_token):
        headers = {
            'Authorization': f"Bearer {access_token}"
        }
        response = requests.get(API_BASE_URL + 'me/playlists', headers=headers)
        if response.status_code == 200:
            return response.json().get('items', [])
        else:
            st.error("Failed to fetch user's playlists. Error: " + response.text)
            return []

    def get_playlist_tracks(access_token, playlist_id):
        headers = {
            'Authorization': f"Bearer {access_token}"
        }
        response = requests.get(API_BASE_URL + f'playlists/{playlist_id}/tracks', headers=headers)
        if response.status_code == 200:
            return response.json().get('items', [])
        else:
            st.error("Failed to fetch playlist tracks. Error: " + response.text)
            return []

    def get_track_preview_url(access_token, track_id):
        headers = {
            'Authorization': f"Bearer {access_token}"
        }
        response = requests.get(API_BASE_URL + f'tracks/{track_id}', headers=headers)
        if response.status_code == 200:
            return response.json().get('preview_url')
        else:
            st.error("Failed to fetch track preview URL. Error: " + response.text)
            return None

    def display_playlist_tracks(tracks):
        for index, track in enumerate(tracks, start=1):
            track_name = track['track']['name']
            artist_names = ', '.join([artist['name'] for artist in track['track']['artists']])
            st.write(f"{index}. {track_name} - {artist_names}")

    # Get access token (you can replace this with your own method of obtaining the access token)
    access_token = st.session_state.access_token

    # Fetch user's playlists
    playlists = get_user_playlists(access_token)

    # Display playlists
    selected_playlist_index = st.selectbox("Select a playlist", range(len(playlists)), format_func=lambda i: playlists[i]['name'])

    if selected_playlist_index is not None:
        selected_playlist_id = playlists[selected_playlist_index]['id']
        st.write(f"Selected Playlist: {playlists[selected_playlist_index]['name']}")

        # Fetch tracks of the selected playlist
        playlist_tracks = get_playlist_tracks(access_token, selected_playlist_id)

        # Display tracks of the selected playlist
        st.write("Playlist Tracks:")
        display_playlist_tracks(playlist_tracks)

        # Add selected tracks to the queue
        st.write("Add songs to queue:")
        selected_tracks = st.multiselect("Select songs to add to queue", [track['track']['name'] for track in playlist_tracks])

        if selected_tracks:
            st.write("Songs added to queue:")
            for track_name in selected_tracks:
                st.write(track_name)

            # Fetch preview URLs for selected tracks and create audio player
            audio_urls = []
            for track in playlist_tracks:
                if track['track']['name'] in selected_tracks:
                    preview_url = get_track_preview_url(access_token, track['track']['id'])
                    if preview_url:
                        audio_urls.append(preview_url)

            if audio_urls:
                st.write("Audio Player:")
                queue_index = 0
                audio_player = st.audio(audio_urls[queue_index], format='audio/mp3')

                # Play next song when the forward button is clicked
                if st.button("Forward"):
                    queue_index = (queue_index + 1) % len(audio_urls)
                    audio_player.audio(audio_urls[queue_index], format='audio/mp3')
            else:
                st.warning("No preview URLs found for selected tracks.")

else:
    st.error("Please go to the HOME page and log in")