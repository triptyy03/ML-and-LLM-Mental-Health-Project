
# Mental Health Disorder Diagnosis using Machine Learning & LLM

## 📌 Project Overview
This project aims to bridge the gap between clinical psychology and artificial intelligence by developing a diagnostic pipeline for mental health disorders. Using a dataset of patient symptoms, the project implements **Binary Classification** (Screening: Normal vs. Disorder) and **Multiclass Classification** (Diagnosis: Normal, Bipolar Type-1, Bipolar Type-2, and Depression).

We compare traditional Machine Learning (LR, KNN, RF)  and explore the reasoning capabilities of Large Language Models (LLMs).

## 📊 Dataset Description
The dataset consists of patient records with several clinical features:
- **Ordinal Features:** Sadness, Euphoria, Exhaustion, Sleep Disorder (Mapped 1-4).
- **Binary Symptoms:** Mood Swings, Suicidal Thoughts, Anorexia, Overthinking, etc.
- **Scale Features:** Optimism, Concentration, Sexual Activity (Scale 1-10).
- **Target:** Expert Diagnosis (Normal, Bipolar Type-1, Bipolar Type-2, Depression).

## 🛠️ Tech Stack
- **Languages:** Python 3.x
- **Libraries:** Pandas, NumPy, Scikit-Learn, TensorFlow/Keras, Seaborn, Matplotlib.
- **Models:** - Logistic Regression (LR)
  - K-Nearest Neighbors (KNN)
  - Random Forest (RF)
  - LLM Prompt Engineering (Logic for zero-shot clinical reasoning)

## 🚀 Key Features & Methodology
1. **Preprocessing Pipeline:**
   - Strategic mapping of categorical text to numerical values.
   - **Feature Scaling:** Implemented `StandardScaler` with a strict split-before-fit approach to prevent data leakage.
2. **Model Optimization:** - Used `GridSearchCV` and `StratifiedKFold` for hyperparameter tuning.
3. **Evaluation Metrics:**
   - Multi-metric analysis using Accuracy, Weighted F1-Score, and Confusion Matrices.
4. **Explainable AI (LLM):**
   - Developed a "Textualization" function to convert tabular rows into clinical narratives for LLM reasoning.

## 📈 Performance Summary
The models were evaluated on their ability to differentiate between complex mood disorders:
- **Binary RF:** Achieved high accuracy in identifying general mental health risk.
- **Multiclass RF:** Showed the best balance in distinguishing Bipolar types vs. Depression.


## 📂 Project Structure
```text
├── LR_KNN_RF_Binary_Multiclass.ipynb  # Full Implementation & Comparison
├── Dataset-Mental-Disorders.csv       # Clinical Dataset
├── README.md                          # Project Documentation
└── plots/                             # Confusion matrices and importance charts
