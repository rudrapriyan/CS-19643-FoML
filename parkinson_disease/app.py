import streamlit as st
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode



def load_audio(file):
    y, sr = librosa.load(file)
    return y, sr



def extract_mfcc(y, sr, n_mfcc=22):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs.T 



def process_audio_from_bytes(audio_bytes):
    y, sr = librosa.load(audio_bytes)
    mfccs = extract_mfcc(y, sr)
    
  
    max_time_steps = 100
    if mfccs.shape[0] < max_time_steps:
        padded_mfcc = np.pad(mfccs, ((0, max_time_steps - mfccs.shape[0]), (0, 0)), mode='constant')
    else:
        padded_mfcc = mfccs[:max_time_steps]
        return padded_mfcc.flatten()

def train_model(features, labels):
    X_flattened = np.array([f.flatten() for f in features])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flattened)
    
    model = RandomForestClassifier()
    model.fit(X_scaled, labels)
    return model, scaler

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_data = None

    def recv(self, frame):
        self.audio_data = frame
        return frame


def main():
    st.title("Parkinson's Disease Detection from Voice Audio")

    st.sidebar.header("Audio Recording")
    st.write("You can now record your voice to predict the presence of Parkinson's Disease.")
    
    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=AudioProcessor,
        async_processing=True,
    )

    if webrtc_ctx.audio_processor and webrtc_ctx.audio_processor.audio_data:
        audio_bytes = webrtc_ctx.audio_processor.audio_data.to_ndarray()
        
        features = []
        labels = [] 
        
        features.append(process_audio_from_bytes(audio_bytes))
        labels.append(1) 
        model, scaler = train_model(features, labels)
        
        # Predict
        X_test = np.array([process_audio_from_bytes(audio_bytes)])
        X_test_scaled = scaler.transform(X_test.flatten().reshape(1, -1))
        prediction = model.predict(X_test_scaled)
        
        if prediction[0] == 1:
            st.write("Prediction: The person might have Parkinson's Disease.")
        else:
            st.write("Prediction: The person does not have Parkinson's Disease.")
    

if __name__ == "__main__":
    main()
