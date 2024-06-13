import streamlit as st
import requests
from datetime import datetime
API_BASE_URL = 'https://api.spotify.com/v1/'
st.set_page_config(page_title="Genre", page_icon="🎧" )
st.title("Songs by Genre")

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

    def get_playlist_tracks(playlist_id):
        if 'access_token' not in st.session_state:
            st.warning("Please log in to Spotify.")
            return

        headers = {
            'Authorization': f"Bearer {st.session_state.access_token}"
        }

        # Make a request to get the tracks from the playlist
        response = requests.get(API_BASE_URL + f'playlists/{playlist_id}/tracks', headers=headers)
        playlist_tracks = response.json()   
        return playlist_tracks 

    def get_current_user_followed_artist():
        headers = {
            'Authorization': f"Bearer {st.session_state.access_token}"
        }
        # Get followed artists
        followed_artists_response = requests.get(API_BASE_URL + 'me/following?type=artist', headers=headers)
        followed_artists = followed_artists_response.json()
        return followed_artists

    def get_artist_details(artist_id):
        headers = {
            'Authorization': f"Bearer {st.session_state.access_token}"
        }

        # Make a request to get the artist details
        response = requests.get(API_BASE_URL + f'artists/{artist_id}', headers=headers)
        artist_details = response.json()
        return artist_details


    def get_playlists():
        headers = {
            'Authorization': f"Bearer {st.session_state.access_token}"
        }

        response = requests.get(API_BASE_URL + 'me/playlists', headers=headers)
        playlists = response.json()
        return playlists

    def get_current_user_followed_artist():
        headers = {
            'Authorization': f"Bearer {st.session_state.access_token}"
        }
        # Get followed artists
        followed_artists_response = requests.get(API_BASE_URL + 'me/following?type=artist', headers=headers)
        followed_artists = followed_artists_response.json()   
        return followed_artists

    #Songs by genre
    def songs_by_genre(desired_genre, limit):
        limit1 = int(limit.split('-')[0])
        playlists = get_playlists()
        k=0
        for playlist in playlists['items']:       
            playlist_id = playlist['id']
            k=k+1
            if k==limit1:
                break
        songs= get_playlist_tracks(playlist_id)

        tracks = []
        for item in songs['items']:
            track = item['track']
            track_name = track['name']
            artist_name = track['artists'][0]['name']
            # Get the first artist of the track
            artist = get_artist_details(track['artists'][0]['id'])
            
            # Get the genres associated with the artist
            genres = artist['genres'] if 'genres' in artist else []
            
            #desired_genre = ''#for streamlit make this user dependent by providing options
            if desired_genre in genres:                   
                tracks.append(f"{track_name} by {artist_name} - (Genre: {genres}) ")
        
        for track0 in tracks:
            if track0 is not None:
                st.write(track0)
                for item in songs['items']:
                    track = item['track']
                    if track['name'] == track0.split(' by')[0]:
                        preview_url = track['preview_url'] if 'preview_url' in track else None
                        # Display the audio player if a preview URL is available
                        if preview_url:
                                st.audio(preview_url)
                        else:
                                st.write("Preview not available")
            
    def search_for_genre():
        headers = {
        'Authorization': f"Bearer {st.session_state.access_token}",
        'Accept': 'application/json',
        'Content-Type': 'application/json'
        }

        select_genre = ['filmi', 'desi pop', 'modern bollywood', 'punjabi pop', 'indian instrumental']
        genre =st.selectbox('Select genre to get songs:', select_genre)


        # Search for songs in the specified genre
        params = {
            'q': f'genre:{genre}',
            'type': 'track',
            'limit': 10
        }
        response = requests.get('https://api.spotify.com/v1/search', params=params, headers=headers)
        data = response.json()
        tracks = data['tracks']['items']

        # Display the list of songs
        if st.button("Get Songs"):
            st.write(f'## Songs in the genre "{genre}"')
            for i, track in enumerate(tracks):
                st.write(f'{i+1}. {track["name"]} - {track["artists"][0]["name"]}')


    # To get songs based on genre:
    playlists = get_playlists()
    playlist_options =[]
    i=1
    if playlists:
        for playlist in playlists['items']:    
            playlist_options.append(f"{i}-{playlist['name']}")
            i=i+1
    st.subheader("To get songs based on genre from your playlist:")
    select_genre = ['filmi', 'desi pop', 'modern bollywood', 'punjabi pop', 'indian instrumental']
    selectP = st.selectbox('Select playlist:', playlist_options)
    select = st.selectbox('Select genre to get the related songs:', select_genre)
    # Your main streamlit app code
    if st.button("Get Songs by Genre"):
        st.success('Button clicked!')    
        tracks0=songs_by_genre(select, selectP)
        for track0 in tracks0:
            st.write(track0)
    
    st.subheader("Get songs of your  favourite genre:")
    search_for_genre()

else:
    st.warning("Visit the localhost site that is opened on new tab if you have logged OR")
    st.error("Please go to the HOME page and log in")
