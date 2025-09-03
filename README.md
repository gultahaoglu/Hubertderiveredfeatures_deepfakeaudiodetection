# HuBERT-Derived SSL Features and ECAPA-TDNN Matching for Robust Audio Deepfake Detection

This repository contains code for the paper **“HuBERT-Derived SSL Features and ECAPA-TDNN Matching for Robust Audio Deepfake Detection”** (MLSP 2025).  
We leverage self-supervised **HuBERT** representations and an **ECAPA-TDNN**-based matching head to build a strong and robust audio deepfake detector. Evaluation follows ASVspoof protocols with **EER** and **min-tDCF**.

---

## Highlights

- **SSL feature extraction:** Intermediate layer outputs from HuBERT (e.g., Large, selectable layer).  
- **Matching head:** ECAPA-TDNN encoder with a similarity-based decision for bona-fide vs. spoofed audio.  
- **Fusion options:**  
  - *Embedding-level fusion* (scripts `2_*`)  
  - *Feature-level fusion* (scripts `3_*`)  
- **Metrics:** EER and min-tDCF (ASVspoof’19 LA protocols).

---


DOI / BibTeX will be added upon camera-ready.
