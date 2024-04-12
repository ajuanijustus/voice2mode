from wav_to_embedding import wav_to_embedding, get_wav_files
import os
import tqdm
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras import layers, models

def generate_embeddings(data_directory, model_name):

    wav_files = get_wav_files(data_directory)

    embeddings = []
    for file_path, label in tqdm.tqdm(wav_files, desc="Processing WAV files"):
        # Obtain the embedding for the current WAV file
        embedding_tensors = wav_to_embedding(file_path, model_name)
        
        # Flatten each layer's embedding and store them separately
        for layer_num, embedding_tensor in enumerate(embedding_tensors):
            embedding = np.mean(embedding_tensor.detach().numpy(), axis=1)[0]
                
            # Append the file name, label, and flattened embedding to the data list
            embeddings.append((file_path, label, layer_num, embedding))

    # Convert the data list to a DataFrame
    embeddings_df = pd.DataFrame(embeddings, columns=['Soundtrack', 'Label', 'Layer', 'Embedding'])

    # Iterate through each layer
    for layer_num in tqdm.tqdm(embeddings_df['Layer'].unique(), desc="Generating embeddings layer-wise"):
        # Clone the original DataFrame to preserve the original embeddings
        phonation_mode_df = embeddings_df[embeddings_df['Layer'] == layer_num].copy(deep=True)
        # Extract embeddings for the current layer
        max_length = max(len(embedding) for embedding in phonation_mode_df['Embedding'])
        X = np.array(list(phonation_mode_df['Embedding']))
        y = phonation_mode_df['Label']

        # Specify the folder path
        folder_path = "embeddings/"+model_name+"/"+str(layer_num)+"/"

        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Save the array in the specified folder
        np.save(folder_path + "X.npy", X)
        np.save(folder_path + "y.npy", y)

def classify_phonation_mode(model_name):

    # Create a DataFrame to store results
    results_df = pd.DataFrame({'Actual_Label': [], 'Predicted_Label': [], 'Layer': [], 'Layer_Accuracy': []})

    # Specify the folder path and model name
    folder_path_base = "embeddings/" + model_name + "/"

    # Get the list of layer folders
    layer_folders = sorted([int(folder) for folder in os.listdir(folder_path_base) if os.path.isdir(os.path.join(folder_path_base, folder))])

    # Iterate over the layer folders
    for layer_num in tqdm.tqdm(layer_folders, desc="Layer-wise classification"):
        # Specify the folder path for the current layer
        folder_path = os.path.join(folder_path_base, str(layer_num))

        # Load X and y from the corresponding files
        X = np.load(os.path.join(folder_path, "X.npy"), allow_pickle=True)
        y = np.load(os.path.join(folder_path, "y.npy"), allow_pickle=True)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train SVM classifier
        svm_classifier = SVC(kernel='linear')  # Linear kernel works well for high-dimensional data
        svm_classifier.fit(X_train, y_train)

        # Evaluate classifier
        y_pred = svm_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        # print(f"Accuracy for Layer {layer_num}: {accuracy}")

        # Create a DataFrame to store results for the current layer
        layer_results_df = pd.DataFrame({
            'Actual_Label': y_test,
            'Predicted_Label': y_pred,
            'Layer': layer_num,
            'Layer_Accuracy': accuracy
        })
        
        # Concatenate the results with the overall results_df
        results_df = pd.concat([results_df, layer_results_df], ignore_index=True)

    return results_df


def k_fold_cross_val(model_name, k=5):
    # Create a DataFrame to store results
    results_df = pd.DataFrame({'Layer': [], 'Fold': [], 'Fold_Accuracy': []})
    confusion_matrices = []

    # Specify the folder path and model name
    folder_path_base = "embeddings/" + model_name + "/"

    # Get the list of layer folders
    layer_folders = sorted([int(folder) for folder in os.listdir(folder_path_base) if os.path.isdir(os.path.join(folder_path_base, folder))])

    # Iterate over the layer folders
    for layer_num in tqdm.tqdm(layer_folders, desc="Layer-wise classification"):
        # Specify the folder path for the current layer
        folder_path = os.path.join(folder_path_base, str(layer_num))

        # Load X and y from the corresponding files
        X = np.load(os.path.join(folder_path, "X.npy"), allow_pickle=True)
        y = np.load(os.path.join(folder_path, "y.npy"), allow_pickle=True)

        # Initialize StratifiedKFold
        skf = StratifiedKFold(n_splits=k)

        # Store cross-validation results
        for fold_num, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Train SVM classifier
            svm_classifier = SVC(kernel='linear')  # Linear kernel works well for high-dimensional data
            svm_classifier.fit(X_train, y_train)

            # Calculate accuracy
            accuracy = svm_classifier.score(X_test, y_test)

            # Print accuracy for each fold
            # print(f"Accuracy for Layer {layer_num}, Fold {fold_num}: {accuracy}")

            # Store fold results
            fold_results_df = pd.DataFrame({
                'Layer': [layer_num],
                'Fold': [fold_num],
                'Fold_Accuracy': [accuracy]
            }, index=[0])
            results_df = pd.concat([results_df, fold_results_df], ignore_index=True)

            # Generate confusion matrix
            y_pred = svm_classifier.predict(X_test)
            class_labels = ["Flow", "Breathy", "Pressed", "Normal"]
            cm = confusion_matrix(y_test, y_pred)
            confusion_matrices.append((layer_num, fold_num, cm))

    return results_df, confusion_matrices


def svm_hyperparameter_tuning(model_name, layer, param_grid, cv=5):

    folder_path = "embeddings/" + model_name + "/" + str(layer) + "/"
    # Load X and y from the corresponding files
    X = np.load(os.path.join(folder_path, "X.npy"), allow_pickle=True)
    y = np.load(os.path.join(folder_path, "y.npy"), allow_pickle=True)

    svm_classifier = SVC(kernel='linear')
    grid_search = GridSearchCV(svm_classifier, param_grid, cv=cv)
    grid_search.fit(X, y)
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    return best_params, best_score

def k_fold_cross_val_xgb(model_name, k=5):

    # Create a DataFrame to store results
    results_df = pd.DataFrame({'Layer': [], 'Fold': [], 'Fold_Accuracy': []})

    # Specify the folder path and model name
    folder_path_base = "embeddings/" + model_name + "/"

    # Get the list of layer folders
    layer_folders = sorted([int(folder) for folder in os.listdir(folder_path_base) if os.path.isdir(os.path.join(folder_path_base, folder))])

    # Iterate over the layer folders
    for layer_num in tqdm.tqdm(layer_folders, desc="Layer-wise classification"):
        # Specify the folder path for the current layer
        folder_path = os.path.join(folder_path_base, str(layer_num))

        # Load X and y from the corresponding files
        X = np.load(os.path.join(folder_path, "X.npy"), allow_pickle=True)
        y = np.load(os.path.join(folder_path, "y.npy"), allow_pickle=True)

        # Label Encoding for y
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        # Initialize StratifiedKFold
        skf = StratifiedKFold(n_splits=k)

        # Store cross-validation results
        for fold_num, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Train XGBoost classifier
            xgb_classifier = xgb.XGBClassifier()
            xgb_classifier.fit(X_train, y_train)

            # Calculate accuracy
            accuracy = xgb_classifier.score(X_test, y_test)
            
            # Print accuracy for each fold
            # print(f"Accuracy for Layer {layer_num}, Fold {fold_num}: {accuracy}")

            # Store fold results
            fold_results_df = pd.DataFrame({
                'Layer': [layer_num],
                'Fold': [fold_num],
                'Fold_Accuracy': [accuracy]
            }, index=[0])
            results_df = pd.concat([results_df, fold_results_df], ignore_index=True)

    return results_df

def create_nn_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=input_shape))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def k_fold_cross_val_nn(model_name, k=5):
    # Create a DataFrame to store results
    results_df = pd.DataFrame({'Layer': [], 'Fold': [], 'Fold_Accuracy': []})

    # Specify the folder path and model name
    folder_path_base = "embeddings/" + model_name + "/"

    # Get the list of layer folders
    layer_folders = sorted([int(folder) for folder in os.listdir(folder_path_base) if os.path.isdir(os.path.join(folder_path_base, folder))])

    # Iterate over the layer folders
    for layer_num in tqdm.tqdm(layer_folders, desc="Layer-wise classification"):
        # Specify the folder path for the current layer
        folder_path = os.path.join(folder_path_base, str(layer_num))

        # Load X and y from the corresponding files
        X = np.load(os.path.join(folder_path, "X.npy"), allow_pickle=True)
        y = np.load(os.path.join(folder_path, "y.npy"), allow_pickle=True)

        # Label Encoding for y
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        # Initialize StratifiedKFold
        skf = StratifiedKFold(n_splits=k)

        # Store cross-validation results
        for fold_num, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Create fully connected neural network model
            input_shape = (X_train.shape[1],)
            num_classes = len(label_encoder.classes_)
            nn_model = create_nn_model(input_shape, num_classes)

            # Train the model
            nn_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

            # Evaluate the model
            accuracy = nn_model.evaluate(X_test, y_test, verbose=0)[1]

            # Print accuracy for each fold
            # print(f"Accuracy for Layer {layer_num}, Fold {fold_num}: {accuracy}")

            # Store fold results
            fold_results_df = pd.DataFrame({
                'Layer': [layer_num],
                'Fold': [fold_num],
                'Fold_Accuracy': [accuracy]
            }, index=[0])
            results_df = pd.concat([results_df, fold_results_df], ignore_index=True)

    return results_df

