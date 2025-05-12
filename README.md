# 🌊 Spatially Transferable Hydrographic Feature Delineation using Meta-Learning and IfSAR Data

This repository contains code, models, and documentation for the research project **"Spatially Transferable Hydrographic Feature Delineation from IfSAR Data: A Meta-Learning Approach."** The project demonstrates the use of Model-Agnostic Meta-Learning (MAML) to improve the transferability of hydrographic feature extraction across diverse terrain conditions in Alaska using high-resolution IfSAR data.

## 🧠 Highlights

- **Meta-Learning Framework**  
  Implements a MAML-based approach to fine-tune U-Net models on new watersheds with limited labeled data.

- **Multimodal Inputs**  
  Uses 5-meter resolution IfSAR-derived datasets including:
  - Digital Terrain Model (DTM)  
  - Digital Surface Model (DSM)  
  - Orthorectified Radar Intensity (ORI)  
  - Derived geomorphometric layers (e.g., curvature, TPI, openness)

- **Episodic Training**  
  Supports few-shot training across grouped watersheds for spatial generalization.

- **Extensibility and Efficiency**  
  Includes scripts for training, adapting, and evaluating models on unseen clusters.

## 📂 Repository Structure

```
├── data/                   # the data for the experiments
├── libs/                   # utitlity functions and files 
├── run_experiments/        # bash script that will run the experiments
├── requirements.txt        # Python dependencies
└── README.md               # Project overview
```

Data can be provided on request.