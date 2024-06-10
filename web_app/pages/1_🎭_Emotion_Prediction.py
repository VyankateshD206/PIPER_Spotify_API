import torch
import streamlit as st
import librosa
import numpy as np
from st_audiorec import st_audiorec
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.signal import stft
import librosa.display
import torch.nn as nn
st.set_page_config(page_title="Emotion Prediction", page_icon="🎭" )

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
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

# Convert emotions into labels
e_label = {'calm': 0, 'happy': 1, 'angry': 2, 'disgust': 3, }

class EmotionRecognizer(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output):
        super(EmotionRecognizer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size3, output)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)       
        return out

checkpoint = torch.load('pytorch_model1.pth')
model = EmotionRecognizer(checkpoint['metadata']['input_size'], checkpoint['metadata']['hidden_size1'],checkpoint['metadata']['hidden_size2'],checkpoint['metadata']['hidden_size3'], checkpoint['metadata']['output'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Function to extract features from user-input voice
def extract_features_user_input(y, sr):
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    return np.hstack((mfccs, chroma, mel))

def plot_spectrogram(Y, sr, hop_length, y_axis="linear"):
        plt.figure(figsize=(25, 10))
        librosa.display.specshow(Y, 
                                sr=sr, 
                                hop_length=hop_length, 
                                x_axis="time", 
                                y_axis=y_axis)
        plt.colorbar(format="%+2.f")
        st.pyplot()

# Streamlit app
def main():
    st.title("Emotion Prediction from Voice")
    st.write(" # Team :green[PIPER]")

    wav_audio_data=st_audiorec()

    if wav_audio_data is not None:
        st.audio(wav_audio_data, format="audio/wav")

    uploaded_file = st.file_uploader("Upload a voice file (in WAV format):", type=["wav"])

    FRAME_SIZE = 2048
    HOP_SIZE = 512

    if uploaded_file is not None:
      
        y, sr = librosa.load(uploaded_file, sr=None)
        features = extract_features_user_input(y, sr)
        input_tensor = torch.FloatTensor(features)
        input_tensor = input_tensor.unsqueeze(0)
        with torch.no_grad():
            model_output = model(input_tensor)

        predicted_label_index = torch.argmax(model_output, dim=1).item()
        predicted_emotion = [emotion for emotion, label in e_label.items() if label == predicted_label_index][0]

        st.success(f"Predicted Emotion: {predicted_emotion}")
        st.audio(uploaded_file)
        S_scale = librosa.stft(y=y, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
        Y_scale = np.abs(S_scale) ** 2
        plot_spectrogram(Y_scale, sr, HOP_SIZE)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        
if __name__ == "__main__":
    main()
