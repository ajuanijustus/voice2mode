# voice2mode: Singing Phonation Mode Classification with Speech Foundation Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**voice2mode** is an open-source framework for automatic classification of singing phonation modes—breathy, neutral (modal), flow, and pressed—using embeddings from large self-supervised speech models (HuBERT, wav2vec2.0). This repository accompanies the peer-reviewed paper:

> **voice2mode: Singing Phonation Mode Classification with Speech Foundation Models**  
> Aju Ani Justus, Ruchit Agrawal, Sudarsana Reddy Kadiri, Shrikanth Narayanan, ICASSP Workshop Proceedings (Speech, Music and Mind), 2026
> [doi.org/10.1109/ICASSP55912.2026.11460695](http://doi.org/10.1109/ICASSP55912.2026.11460695)


## Overview

Traditional approaches to singing phonation mode classification rely on handcrafted features or task-specific neural networks. **voice2mode** demonstrates that representations from speech foundation models, pre-trained on large speech corpora, can be effectively transferred to singing voice analysis—substantially outperforming conventional spectral features.

**Key Features:**
- Extracts layer-wise embeddings from HuBERT and wav2vec2.0 (Base & Large).
- Applies global temporal pooling to obtain fixed-size feature vectors.
- Classifies phonation mode using lightweight classifiers (SVM, XGBoost).
- Benchmarks against standard baselines (spectrogram, mel-spectrogram, MFCC).
- Reproducible experiments on a public soprano singing dataset.


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results](#results)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)


## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ajuanijustus/voice2mode.git
   cd voice2mode
   ```

2. **Set up the environment:**
   We recommend using Python 3.10 via Conda to ensure smooth library integration (especially for macOS users running XGBoost):
   ```bash
   conda create -n voice2mode python=3.10 -y
   conda activate voice2mode
   conda install -c conda-forge xgboost -y
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   Main dependencies include PyTorch, TensorFlow, Transformers, SoundFile, Librosa, Scikit-Learn, and Jupyter Notebook.


## Usage

### 1. Prepare the Dataset

Download the [Soprano Phonation Modes Dataset](http://www.proutskova.de/phonation-modes/) and place the audio files in the `data/` directory (see [Dataset](#dataset) for details), or you can use your own voice dataset.

📬 Need help? If you run into issues formatting or structuring your dataset, feel free to mail me at research@ajuanijustus.com!

### 2. Feature Extraction and Classification

Instead of command-line scripts, the entire pipeline is executed sequentially via Jupyter Notebooks.
Open Jupyter:
```bash
jupyter notebook
```
Run notebooks 1 through 4 to handle feature extraction, embedding generation, and baseline/advanced classification models. Running these notebooks will automatically create a `results/` directory. All generated classification reports, CSV logs, and visual performance charts (e.g., confusion matrices) are cleanly outputted directly to `results/`.

Supported classifiers: `svm`, `xgboost`.
Supported models: `hubert`, `wav2vec2-base`, `wav2vec2-large`.

### 3. Evaluate Models and Classifiers

To evaluate all models side-by-side and completely reproduce the main results, tables, and figures from the paper, execute the final evaluation notebook: `5_evaluation.ipynb`.

### 4. Repository Architecture and Helper Functions

The standalone Python files in the root folder act as core utility scripts behind the scenes. If you want to dive deeper, tweak the pipeline, or look for optimizations, feel free to explore them:
1. `wav_to_embedding.py`: Handles loading audio files and passing them through deep self-supervised models (transformers) to extract rich audio embedding vectors.
2. `baseline_features.py`: Manages classical audio DSP feature extraction (MFCCs, deltas, spectral features) via librosa.
3. `classifier.py`: Contains the architectures and training logic for the downstream classifiers (SVM, XGBoost, and Neural Networks).

## Dataset

We use the [Soprano Phonation Modes Dataset](http://www.proutskova.de/phonation-modes/) ([Proutskova et al., 2013](#acknowledgements)), which contains 763 sustained vowel recordings by a professional soprano, labeled as breathy, modal, flow, or pressed.  
- **Vowels:** 9 types  
- **Pitch range:** A3–G5  
- **Sampling rate:** 44.1 kHz (downsampled to 16 kHz for experiments)

**Note:** The dataset is publicly available under a Creative Commons license. Please cite the original authors if you use the data.


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


## Reproducibility

- All experiments can be reproduced end-to-end using the sequential notebooks.
- Results are validated using a rigorous 5-fold stratified cross-validation setup.
- For archive purposes, an unmaintained notebook (`archive/rl_classification_archive.ipynb`) exploring Reinforcement Learning for classification is retained in the repository for posterity.

## Citation

If you use this code or finding from the paper, please cite:

```
@inproceedings{voice2mode,
  author={Justus, Aju Ani and Agrawal, Ruchit and Kadiri, Sudarsana Reddy and Narayanan, Shrikanth},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={voice2mode: Phonation Mode Classification in Singing Using Self-Supervised Speech Models}, 
  year={2026},
  pages={22072-22076},
  doi={10.1109/ICASSP55912.2026.11460695}
}
```


## License

This repository is licensed under the [MIT License](LICENSE).


## Acknowledgements

- Soprano Phonation Modes Dataset: [Proutskova et al., 2013](http://www.proutskova.de/phonation-modes/)
- Pre-trained models: [wav2vec2.0](https://github.com/pytorch/fairseq), [HuBERT](https://github.com/pytorch/fairseq)

---

For questions, collaboration, or contributions, please open an issue, submit a pull request, or drop me an email at research@ajuanijustus.com.
