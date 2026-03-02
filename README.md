# Polito_multix
## Extended HydraFusion for E‑Health Waveform Prediction

This repository extends the original HydraFusion model  
(https://github.com/AICPS/hydrafusion)  
to support waveform prediction and multimodal fusion for e‑health applications.

### Code Extensions
- `stem.py`  
  - Added stems for **heatmaps** and **spectrograms**
- `branch.py`  
  - Added branches for **heatmaps**, **spectrograms**, and **early‑fusion**
- `hydranet.py`  
  - Extended to support **waveform regression**

---

## Datasets

### BS‑Breath Dataset
Source: https://gitlab.kuleuven.be/u0149002/bs-breath/-/tree/main?ref_type=heads

Preprocessed data included:
- `rf_heatmaps.jpg`
- `rf_spectogram.jpg`

MATLAB preprocessing scripts:  
`matlab_preprocessing/`

### MIT Sleep Database (SLPDB)
Source: https://physionet.org/content/slpdb/1.0.0/

Preprocessed data included:
- `ecg_heatmap.png`
- `eeg_spectrogram.png`
- `resp_segment/` (RGB respiratory segments)

Preprocessing scripts:  
`mit_preprocessing/`

---

## Training & Evaluation Pipelines

### BS‑Breath
- `rf_wave.py` — centralized training & evaluation  
- `split_train_rf.py` — federated training & evaluation

### MIT Dataset
- `mit_wave.py` — centralized training & evaluation  
- `split_train_mit.py` — federated training & evaluation
# Initial model outputs
- mit_prediction.png and mit_prediction_federated.png - the first-pass results.
- The model still requires additional training and refinement for improved performance.

---


