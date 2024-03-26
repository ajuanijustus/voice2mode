from transformers import AutoProcessor, HubertModel
import soundfile as sf
import numpy as np
import pandas as pd
import librosa
import os

processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

def get_wav_files(directory, debug=False):
    wav_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                # Extract the label (subfolder name)
                label = os.path.basename(root)
                # Construct the full path to the WAV file
                file_path = os.path.join(root, file)
                # Append tuple containing file path and label to list
                wav_files.append((file_path, label))
    if debug:
        # Print file paths and corresponding labels
        for file_path, label in wav_files:
            print(f"File Path: {file_path}, Label: {label}")
    return wav_files

def resample_audio(audio, original_sr, target_sr):
    return librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)

def map_to_array(file_path):
    track, sample_rate = sf.read(file_path)
    if sample_rate != 16000:  # If the sample rate is not 16 kHz, resample it
        track = resample_audio(track, sample_rate, 16000)
        sample_rate = 16000
    return track, sample_rate

def wav_to_embedding(file_path):
    # Load each WAV file, map it to an array and its sample rate
    track, sample_rate = map_to_array(file_path)

    # Preprocess each array and convert it into input values
    input_value = processor(track, sampling_rate=sample_rate, return_tensors="pt").input_values

    # Pass the input values through the model to get the hidden states
    hidden_state = model(input_value).last_hidden_state

    return hidden_state