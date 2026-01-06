# voice2mode: Singing Phonation Mode Classification with Speech Foundation Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**voice2mode** is an open-source framework for automatic classification of singing phonation modes—breathy, neutral (modal), flow, and pressed—using embeddings from large self-supervised speech models (HuBERT, wav2vec2.0). This repository accompanies the peer-reviewed paper:

> **voice2mode: Singing Phonation Mode Classification with Speech Foundation Models**  
> Aju Ani Justus, Ruchit Agrawal, Sudarsana Reddy Kadiri, Shrikanth Narayanan, ICASSP Workshop Proceedings (Speech, Music and Mind), 2026
> [link here after the paper is out]()


## Overview

Traditional approaches to singing phonation mode classification rely on handcrafted features or task-specific neural networks. **voice2mode** demonstrates that representations from speech foundation models, pre-trained on large speech corpora, can be effectively transferred to singing voice analysis—substantially outperforming conventional spectral features.

**Key Features:**
- Extracts layer-wise embeddings from HuBERT and wav2vec2.0 (Base & Large).
- Applies global temporal pooling to obtain fixed-size feature vectors.
- Classifies phonation mode using lightweight classifiers (SVM, XGBoost).
- Benchmarks against standard baselines (spectrogram, mel-spectrogram, MFCC).
- Reproducible experiments on a public soprano singing dataset.

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results](#results)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ajuanijustus/voice2mode.git
   cd voice2mode
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Main dependencies include:
   - Python 3.8+
   - PyTorch
   - torchaudio
   - scikit-learn
   - xgboost
   - librosa
   - numpy, pandas, matplotlib

---

## Usage

### 1. **Prepare the Dataset**

Download the [Soprano Phonation Modes Dataset](http://www.proutskova.de/phonation-modes/) and place the audio files in the `data/` directory. See [Dataset](#dataset) for details.

### 2. **Feature Extraction**

Extract embeddings from pre-trained models:
```bash
python extract_features.py --model hubert --input_dir data/ --output_dir features/
```
Supported models: `hubert`, `wav2vec2-base`, `wav2vec2-large`.

### 3. **Train and Evaluate Classifiers**

Run classification experiments:
```bash
python classify.py --features_dir features/ --classifier svm
```
Supported classifiers: `svm`, `xgboost`.

### 4. **Reproduce Paper Results**

To reproduce the main results and tables from the paper:
```bash
python run_experiments.py --config configs/paper.yaml
```

---

## Dataset

We use the [Soprano Phonation Modes Dataset](http://www.proutskova.de/phonation-modes/) ([Proutskova et al., 2013](#acknowledgements)), which contains 763 sustained vowel recordings by a professional soprano, labeled as breathy, modal, flow, or pressed.  
- **Vowels:** 9 types  
- **Pitch range:** A3–G5  
- **Sampling rate:** 44.1 kHz (downsampled to 16 kHz for experiments)

**Note:** The dataset is publicly available under a Creative Commons license. Please cite the original authors if you use the data.

---

## Results

**voice2mode** achieves state-of-the-art accuracy for phonation mode classification:

| Feature Type      | SVM Accuracy (%) | XGBoost Accuracy (%) |
|-------------------|------------------|----------------------|
| Spectrogram       | 79.9 ± 2.6       | 79.6 ± 3.1           |
| wav2vec2-BASE     | 90.7 ± 5.1       | 83.7 ± 7.0           |
| wav2vec2-LARGE    | 90.2 ± 5.3       | 82.6 ± 5.1           |
| **HuBERT**        | **95.7 ± 3.0**   | **92.0 ± 4.0**       |

- **Best performance:** HuBERT embeddings (early layers) + SVM
- **Absolute improvement:** ~12–15% over best baseline

See the [paper](#citation) for detailed results, layer-wise analysis, and confusion matrices.

---

## Reproducibility

- All experiments can be reproduced using the provided scripts and configuration files.
- Results are based on 5-fold stratified cross-validation.
- See `configs/` for example experiment setups.

---

## Citation

If you use this code or dataset, please cite:

> fill with bibtex after publication

---

## License

This repository is licensed under the [MIT License](LICENSE).

---

## Acknowledgements

- Soprano Phonation Modes Dataset: [Proutskova et al., 2013](http://www.proutskova.de/phonation-modes/)
- Pre-trained models: [wav2vec2.0](https://github.com/pytorch/fairseq), [HuBERT](https://github.com/pytorch/fairseq)

---

For questions or contributions, please open an issue or pull request.
