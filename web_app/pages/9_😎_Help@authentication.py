import streamlit as st
from requests_oauthlib import OAuth2Session # type: ignore
from requests.auth import HTTPBasicAuth
import requests
import json

AUTH_URL = 'https://accounts.spotify.com/authorize'
TOKEN_URL = 'https://accounts.spotify.com/api/token'
REDIRECT_URI = 'https://piper-spotify.streamlit.app/'  
API_BASE_URL = 'https://api.spotify.com/v1/'
SCOPE = [
    "user-read-email",
    "playlist-read-collaborative"
]

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

CLIENT_ID = st.secrets["SPOTIPY_CLIENT_ID"]
CLIENT_SECRET = st.secrets["SPOTIPY_CLIENT_SECRET"]

def login():
    spotify = OAuth2Session(CLIENT_ID, scope=SCOPE, redirect_uri=REDIRECT_URI)
    authorization_url, state = spotify.authorization_url(AUTH_URL)
    st.write(f"Click [here]({authorization_url}) to login with Spotify.")

def callback():
    code = st.text_input('Enter the code from the callback URL:')
    if st.button('Submit') and code:
        res = requests.post(TOKEN_URL,
                            auth=HTTPBasicAuth(CLIENT_ID, CLIENT_SECRET),
                            data={
                                'grant_type': 'authorization_code',
                                'code': code,
                                'redirect_uri': REDIRECT_URI
                            })
        st.json(res.json())

def get_current_user_info():
    headers = {
        'Authorization': f"Bearer {st.session_state.access_token}"
    }

    response = requests.get(API_BASE_URL + 'me', headers=headers)
    user_info = response.json()
    st.write(f" #### Logged in as :green({user_info['display_name']})")

def main():
    if 'access_token' in st.session_state:
        st.warning("If you are not logged in using the link at home page then use this to log in!")
        st.write('### Spotify Authorization')
        login()
        callback()
        
        
    else: st.error("Failed to authenticate user. Please authenticate.")


if __name__ == '__main__':
    main()
