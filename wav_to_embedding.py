from transformers import Wav2Vec2Processor, Wav2Vec2Model
import soundfile as sf
import librosa

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = Wav2Vec2Model.from_pretrained("facebook/hubert-large-ls960-ft")

def resample_audio(audio, original_sr, target_sr):
    return librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)

def map_to_array(file_path):
    speech, sample_rate = sf.read(file_path)
    if sample_rate != 16000:  # If the sample rate is not 16 kHz, resample it
        speech = resample_audio(speech, sample_rate, 16000)
        sample_rate = 16000
    return speech, sample_rate

# Assuming wav_files is a list of tuples containing file paths and their corresponding labels
# wav_files = [("path_to_wav_file1.wav", "label1"), ("path_to_wav_file2.wav", "label2"), ...]
wav_files = [('data/Pressed/a4_I_pressed_norm.wav', 'Pressed')]

# Load each WAV file, map it to an array and its sample rate
data = [map_to_array(file_path) for file_path, _ in wav_files]

# Preprocess each array and convert it into input values
input_values = []
for speech, sample_rate in data:
    input_values.append(processor(speech, sampling_rate=sample_rate, return_tensors="pt").input_values)

# Pass the input values through the model to get the hidden states
hidden_states = []
for input_value in input_values:
    hidden_states.append(model(input_value).last_hidden_state)

print(hidden_states)


# wav_files = [('data/Pressed/a4_I_pressed_norm.wav', 'Pressed')]
