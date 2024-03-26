# Phonation Mode Classification using HuBERT Embeddings and SVM Classifier

This Python project aims to classify phonation modes based on the embeddings extracted from the HuBERT foundation model using a Support Vector Machine (SVM) classifier.

## Requirements
- Python 3.x
- PyTorch
- scikit-learn
- HuBERT model (pre-trained or fine-tuned on relevant data)
- Dataset containing labeled phonation mode samples

## Installation

1. Clone the repository:
`git clone https://github.com/ajuanijustus/Breadcrumbsphonation_mode_classifier.git`

2. Install dependencies:
`pip install -r requirements.txt`


## Usage

1. lorem ipsum
2. lorem ipsum
3. lorem ipsum
4. lorem ipsum

## Example

```python
# Example code snippet for using the trained SVM classifier

import numpy as np
from sklearn.svm import SVC

# Load pre-trained SVM classifier
classifier = SVC(kernel='linear')

# Load pre-trained embeddings (X) and corresponding labels (y)
X = np.load('embeddings.npy')
y = np.load('labels.npy')

# Train the classifier
classifier.fit(X, y)

# Now, you can use this classifier to predict phonation modes for new embeddings
```
