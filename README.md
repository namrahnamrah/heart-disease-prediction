# Heart Disease Prediction — SVM vs MLP Neural Network

Binary classification system predicting heart disease risk from clinical patient data, comparing Support Vector Machine (SVM) and Multi-Layer Perceptron (MLP) performance across key diagnostic metrics.

---

## Results

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **SVM (RBF kernel)** | **84.78%** | **84.91%** | **88.24%** | **86.54%** | **0.922** |
| MLP (32→16, ReLU) | 81.52% | 83.33% | 83.33% | 83.33% | 0.856 |

**SVM selected as the stronger model.** AUC-ROC of 0.922 indicates robust discrimination between at-risk and healthy patients across all classification thresholds. In a clinical context, SVM's higher recall (88.24%) is particularly important — it minimises false negatives, the more costly error in disease screening.

---

## Dataset

- **918 patient records**, 20 clinical features
- Features include: age, resting blood pressure, cholesterol, max heart rate, ST depression (oldpeak), exercise-induced angina, and more
- Binary target: presence (1) or absence (0) of heart disease
- Leakage column (`num`) identified and removed during preprocessing
- No missing values; duplicates dropped before modelling

---

## Methodology

### Preprocessing
- Duplicate removal
- Leakage detection and column removal
- Standard scaling (zero mean, unit variance) applied before SVM, MLP, and PCA
- Stratified 80/20 train-test split (`random_state=42`)

### Models
| Model | Configuration |
|-------|--------------|
| SVM | RBF kernel, `probability=True` |
| MLP | Hidden layers: (32, 16), ReLU activation, Adam optimiser, 500 max iterations |

### Evaluation
- Accuracy, Precision, Recall, F1 Score, AUC-ROC
- Confusion matrices for both models
- ROC curve comparison
- PCA (2 components) for exploratory visualisation — 2D class separability check

---

## Project Structure
heart-disease-prediction/
│
├── heart_disease_project.ipynb   # Full pipeline: EDA → modelling → evaluation
├── heart_disease_clean.csv       # Cleaned dataset (918 rows, 22 cols pre-processing)
├── requirements.txt              # Python dependencies
│
└── outputs/
├── confusion_matrix_svm.png
├── confusion_matrix_mlp.png
├── roc_curve_comparison.png
├── pca_scatter.png
├── correlation_heatmap.png
├── numeric_histograms.png
├── boxplots_by_target.png
├── target_class_distribution.png
└── model_comparison.csv

---

## Setup & Usage
pip install -r requirements.txt
jupyter notebook heart_disease_project.ipynb

---

## Key Findings

- **SVM outperforms MLP** on every metric, with the gap most pronounced in AUC-ROC (+0.066) and recall (+4.9%)
- **Recall is the critical metric here** — in disease screening, a false negative (missed diagnosis) carries significantly higher cost than a false positive
- **Class imbalance** in the target was assessed and accounted for; stratified splitting ensured both classes were proportionally represented in train and test sets
- **PCA analysis** confirmed meaningful class separability in 2D, validating that the feature set carries genuine predictive signal

---

## Tech Stack

`Python` · `Scikit-learn` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Jupyter`
