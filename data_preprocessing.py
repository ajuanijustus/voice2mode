import os

def get_wav_files(directory):
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
    return wav_files

data_directory = "data"
wav_files = get_wav_files(data_directory)

# Print file paths and corresponding labels
for file_path, label in wav_files:
    print(f"File Path: {file_path}, Label: {label}")
