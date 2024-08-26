import streamlit as st
import json
from streamlit_lottie import st_lottie
import requests
import urllib.parse
from datetime import datetime
import base64
st.set_page_config(page_title="Home", page_icon="🏠" )

#https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR5Q_Mz2R04N9qHDaW7wpZMCVYdzzE2bZeRMQ&s
page_bg_img=""" 
<style>
.stDeployButton{
    visibility: hidden;
}
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
st.title(f"Welcome to the :green[PIPER] App!")
st.write("Please send your email address to the given email so that we can add you to the allowlist of API ")
st.write("My Email: vyankateshd206@gmail.com")
st.write("If already added then ignore the above message!")

st.write(":brown[Without Login you can use the emotion predictor!]")
#Animations:
def load_lottieurl(url:str):
    r= requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie1= load_lottieurl("https://lottie.host/703aaeb7-0b76-409c-9cc6-6141844695bb/ykIih7POIH.json")

# Set your Spotify API credentials
SPOTIPY_REDIRECT_URI = 'https://piper-spotify.streamlit.app/'
API_BASE_URL = 'https://api.spotify.com/v1/'
scope = 'user-read-private user-read-email playlist-read-private user-follow-read playlist-modify-private playlist-modify-public user-library-read user-top-read '

# with open('client_id.txt') as f:
#     SPOTIPY_CLIENT_ID = f.read()
# with open('client_secret.txt') as f:
#     SPOTIPY_CLIENT_SECRET = f.read()
SPOTIPY_CLIENT_ID = st.secrets["SPOTIPY_CLIENT_ID"]
SPOTIPY_CLIENT_SECRET = st.secrets["SPOTIPY_CLIENT_SECRET"]

def get_current_user_info():
    headers = {
        'Authorization': f"Bearer {st.session_state.access_token}"
    }

    response = requests.get(API_BASE_URL + 'me', headers=headers)
    user_info = response.json()
    st.write(f" #### Logged in as :green({user_info['display_name']})")


query_params =  st.query_params
if "code" in query_params:
    st.session_state.code = query_params["code"]#[0]


if 'refresh_token' not in st.session_state:
    st.session_state['refresh_token']=""

if 'expire_at' not in st.session_state:
    st.session_state['expire_at']=""

params = {
    'client_id': SPOTIPY_CLIENT_ID,
    'response_type': 'code',
    'scope': scope,
    'redirect_uri': SPOTIPY_REDIRECT_URI,
    'show_dialog': True
}

auth_url = f"https://accounts.spotify.com/authorize?{urllib.parse.urlencode(params)}"

# Display link to authenticate
st.header(f"Click [here]({auth_url}) to log in to :green[PIPER] using :green[Spotify] .")

# Retrieve the authentication code from the URL
code = st.session_state.code if "code" in st.session_state else None


if 'access_token' not in st.session_state:
    st.session_state['access_token']= None
    
    
    if code:  
        req_body = {
            'code': code,
            'grant_type': 'authorization_code',
            'redirect_uri': SPOTIPY_REDIRECT_URI,
            'client_id': SPOTIPY_CLIENT_ID,
            'client_secret': SPOTIPY_CLIENT_SECRET
        }

        response = requests.post('https://accounts.spotify.com/api/token', data=req_body)
        token_info = response.json()
        if 'access_token' in token_info:
            # Save the access token in the Streamlit session state
            st.session_state['access_token'] = token_info['access_token']
            st.session_state["refresh_token"] = token_info['refresh_token']
            st.session_state['expire_at'] = datetime.now().timestamp() + token_info['expires_in']
            get_current_user_info()
        else:
            # Show an error message if the 'access_token' key is not found
            st.error("Failed to authenticate user. Please authenticate.")    

st_lottie(
    lottie1,
    speed=1,
    reverse=False,
    loop=True,
    quality='high',
    height=400,
    )

if 'access_token' in st.session_state:
    st.write("#### Select a :red[page] from the :red[sidebar] to get started.")             
st.header("Made by: :green[Vyankatesh Deshpande] | Btech CSE :green[IIT Jodhpur]")
st.image('web_app/spotify_image.png', caption='PIPER', use_column_width=True) 


