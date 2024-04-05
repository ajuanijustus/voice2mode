from wav_to_embedding import wav_to_embedding, get_wav_files
import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

def classify_wav(data_directory, model_name):

    wav_files = get_wav_files(data_directory)

    embeddings = []
    for file_path, label in tqdm.tqdm(wav_files, desc="Processing WAV files"):
        # Obtain the embedding for the current WAV file
        embedding_tensors = wav_to_embedding(file_path, model_name)
        
        # Flatten each layer's embedding and store them separately
        for layer_num, embedding_tensor in enumerate(embedding_tensors):
            embedding = embedding_tensor.detach().numpy().flatten()
            
            if len(embedding) < 80000:
                embedding = np.pad(embedding, (0, 80000 - len(embedding)))
            else:
                embedding = embedding[:80000]
                
            # Append the file name, label, and flattened embedding to the data list
            embeddings.append((file_path, label, layer_num, embedding))

    # Convert the data list to a DataFrame
    embeddings_df = pd.DataFrame(embeddings, columns=['Soundtrack', 'Label', 'Layer', 'Embedding'])

    # Create a DataFrame to store results
    results_df = pd.DataFrame({'Actual_Label': [], 'Predicted_Label': [], 'Layer': []})

    # Iterate through each layer
    for layer_num in tqdm.tqdm(embeddings_df['Layer'].unique(), desc="Layer-wise classification"):
        # Clone the original DataFrame to preserve the original embeddings
        phonation_mode_df = embeddings_df[embeddings_df['Layer'] == layer_num].copy(deep=True)
        # Extract embeddings for the current layer
        max_length = max(len(embedding) for embedding in phonation_mode_df['Embedding'])
        X = np.array(list(phonation_mode_df['Embedding']))
        y = phonation_mode_df['Label']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train SVM classifier
        svm_classifier = SVC(kernel='linear')  # Linear kernel works well for high-dimensional data
        svm_classifier.fit(X_train, y_train)

        # Evaluate classifier
        y_pred = svm_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for Layer {layer_num}: {accuracy}")

        # Create a DataFrame to store results for the current layer
        layer_results_df = pd.DataFrame({
            'Actual_Label': y_test,
            'Predicted_Label': y_pred,
            'Layer': layer_num,
            'Layer_Accuracy': accuracy
        })
        
        # Concatenate the results with the overall results_df
        results_df = pd.concat([results_df, layer_results_df], ignore_index=True)
        
    # Merge results with embeddings_df and fill NA for Predicted_Label
    merged_df = pd.merge(embeddings_df, results_df, left_on=['Layer', 'Label'], right_on=['Layer', 'Actual_Label'], how='left')
    merged_df['Predicted_Label'] = merged_df['Predicted_Label'].fillna(np.nan)

    return merged_df


def k_fold_cross_val(data_directory, model_name, k=5):

    wav_files = get_wav_files(data_directory)
    
    embeddings = []
    for file_path, label in tqdm.tqdm(wav_files, desc="Processing WAV files"):
        # Obtain the embedding for the current WAV file
        embedding_tensors = wav_to_embedding(file_path, model_name)
        
        # Flatten each layer's embedding and store them separately
        for layer_num, embedding_tensor in enumerate(embedding_tensors):
            embedding = embedding_tensor.detach().numpy().flatten()
            
            if len(embedding) < 80000:
                embedding = np.pad(embedding, (0, 80000 - len(embedding)))
            else:
                embedding = embedding[:80000]
                
            # Append the file name, label, and flattened embedding to the data list
            embeddings.append((file_path, label, layer_num, embedding))

    # Convert the data list to a DataFrame
    embeddings_df = pd.DataFrame(embeddings, columns=['Soundtrack', 'Label', 'Layer', 'Embedding'])

    # Create a DataFrame to store results
    results_df = pd.DataFrame({'Layer': []})

    # Iterate through each layer
    for layer_num in tqdm.tqdm(embeddings_df['Layer'].unique(), desc="Layer-wise Cross-validation"):
        # Clone the original DataFrame to preserve the original embeddings
        phonation_mode_df = embeddings_df[embeddings_df['Layer'] == layer_num].copy(deep=True)
        # Extract embeddings for the current layer
        max_length = max(len(embedding) for embedding in phonation_mode_df['Embedding'])
        X = np.array(list(phonation_mode_df['Embedding']))
        y = phonation_mode_df['Label']

        # Train SVM classifier with 5-fold cross-validation
        svm_classifier = SVC(kernel='linear')  # Linear kernel works well for high-dimensional data
        cv_scores = cross_val_score(svm_classifier, X, y, cv=k)

        # Print average accuracy across folds
        print(f"Average accuracy for Layer {layer_num}: {np.mean(cv_scores)}")

        # Store cross-validation results
        for fold_num, accuracy in enumerate(cv_scores):
            fold_results_df = pd.DataFrame({
                'Layer': [layer_num],
                'Fold': [fold_num],
                'Fold_Accuracy': [accuracy]
            }, index=[0])
            results_df = pd.concat([results_df, fold_results_df], ignore_index=True)

    return results_df

from sklearn.model_selection import GridSearchCV

def svm_hyperparameter_tuning(data_directory, model_name, layer, param_grid, cv=5):
    wav_files = get_wav_files(data_directory)
    
    embeddings = []
    for file_path, label in tqdm.tqdm(wav_files, desc="Processing WAV files"):
        # Obtain the embedding for the current WAV file
        embedding_tensors = wav_to_embedding(file_path, model_name)
        
        # Flatten each layer's embedding and store them separately
        for layer_num, embedding_tensor in enumerate(embedding_tensors):
            embedding = embedding_tensor.detach().numpy().flatten()
            
            if len(embedding) < 80000:
                embedding = np.pad(embedding, (0, 80000 - len(embedding)))
            else:
                embedding = embedding[:80000]
                
            # Append the file name, label, and flattened embedding to the data list
            embeddings.append((file_path, label, layer_num, embedding))

    # Convert the data list to a DataFrame
    embeddings_df = pd.DataFrame(embeddings, columns=['Soundtrack', 'Label', 'Layer', 'Embedding'])

    phonation_mode_df = embeddings_df[embeddings_df['Layer'] == layer].copy(deep=True)
    # Extract embeddings for the current layer
    max_length = max(len(embedding) for embedding in phonation_mode_df['Embedding'])
    X = np.array(list(phonation_mode_df['Embedding']))
    y = phonation_mode_df['Label']

    svm_classifier = SVC(kernel='linear')
    grid_search = GridSearchCV(svm_classifier, param_grid, cv=cv)
    grid_search.fit(X, y)
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    return best_params, best_score