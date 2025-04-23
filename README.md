
# Exploring Model Architectures and Modalities for Disease Stage Classification in Amyotrophic Lateral Sclerosis Using Spatial Transcriptomics and Histology Imaging

This repository contains the code and experiments for classifying ALS disease progression stages (p30, p70, p100, p120) using single- and multi-modal models that combine gene expression, histopathology images, and spatial relationships.

## 🔍 Objective
To accurately classify the stages of Amyotrophic Lateral Sclerosis (ALS) by leveraging multimodal data and deep learning techniques, and to evaluate the effectiveness of different model architectures in handling complex disease progression.

## 🧪 Models Implemented
- **MLP (Gene Expression)**
- **Vision Transformer (Histopathology Images)**
- **Hybrid CNN–MLP (Images + Gene)**
- **GNN–CNN–MLP (Spots + Images + Gene)**

## 📊 Performance Metrics
- Accuracy
- True Positive Rate (Sensitivity)
- True Negative Rate (Specificity)
- Precision
- F1-Score
- AUC (Area Under the Curve)
- Cohen’s Kappa
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Relative Absolute Error (RAE)
- Root Relative Squared Error (RRSE)

## 🧬🧫📍 Data
- Spatial transcriptomics gene expression matrix
- Histopathology images of tissue sections
- Spatial location metadata (spot-level)


## 📁 Directory Structure
```
CODE/
├── ALS Classification Code/ # Processed data files, help scripts, Model architecture definitions
├── dataset/                 # Datasets used
├── environment.yml          # Dependencies
└── README.md                # This file
```

## 🛠️ Dependencies
- Python 3.10.16  
- TensorFlow 2.14.1
- NumPy 1.26.4 
- Spektral (for GNN)
- Matplotlib / Seaborn (for plots)
- Pandas / Scikit-learn
- Other dependencies

Install them using:
```bash
conda env create -f environment.yml
```

## 🚀 Results
Multimodal models significantly outperform single-modality ones. The **Hybrid GNN–CNN–MLP** model achieved:
- **97% accuracy**
- Strong sensitivity/specificity across all stages
- High Cohen’s Kappa agreement with ground truth

## 📚 Citation
```
Tan, X., Su, A., Tran, M. and Nguyen, Q., 2020. SpaCell: integrating tissue morphology and spatial gene expression to predict disease cells. Bioinformatics, 36(7), pp.2293-2294.
```

## 🙌 Acknowledgements
- Dataset from: (https://github.com/BiomedicalMachineLearning/Spacell/tree/master/dataset)
- Prof. Claudio Angione / Teesside University, Middlesbrough

## 🧠 Author
**Charles Olanrewaju**
