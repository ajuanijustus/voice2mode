from wav_to_embedding import * 
import tqdm

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data_directory = "unit_test_data"

wav_files = get_wav_files(data_directory)

hubert_embeddings = []
for file_path, label in tqdm.tqdm(wav_files, desc="Processing WAV files"):
    # Obtain the embedding for the current WAV file
    embedding_tensor = wav_to_embedding(file_path)
    
    # Flatten the embedding tensor
    embedding = embedding_tensor.detach().numpy().flatten()  # Convert tensor to numpy array
    
    # Append the file name, label, and flattened embedding to the data list
    hubert_embeddings.append((file_path, label, embedding))

# Convert the data list to a DataFrame
hubert_embeddings_df = pd.DataFrame(hubert_embeddings, columns=['Soundtrack', 'Label', 'Embedding'])

phonation_mode_df = hubert_embeddings_df.copy(deep=True)

# Step 1: Extract features and labels

# X = np.stack(phonation_mode_df['Embedding'])  # Convert string representation of array to numpy array
# y = phonation_mode_df['Label']

# Pad or truncate embeddings to a fixed length
max_length = max(len(embedding) for embedding in phonation_mode_df['Embedding'])
X = np.array([np.pad(embedding, (0, max_length - len(embedding))) for embedding in phonation_mode_df['Embedding']])

# Extract labels
y = phonation_mode_df['Label']

# Step 2: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# normalize the test/train data - ETA to check for variance

# Step 3: Train SVM classifier
svm_classifier = SVC(kernel='linear')  # Linear kernel works well for high-dimensional data
svm_classifier.fit(X_train, y_train)

# Step 4: Evaluate classifier
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Create a DataFrame to store results
results_df = pd.DataFrame({
    'Actual_Label': y_test,
    'Predicted_Label': y_pred,
    'Features': X_test.tolist()
})

results_df.to_csv('results_df.csv', index=False)