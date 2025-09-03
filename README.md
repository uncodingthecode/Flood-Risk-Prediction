# 🌊 Flood Risk Prediction

A data-driven machine learning project to forecast **flood risks in India** using historical rainfall, elevation, and flood history datasets.  
This project leverages multiple ML models to classify regions into **High, Medium, and Low flood risk levels**, enabling proactive disaster management and preparedness.

---

## 📌 Project Overview
Flooding is a major cause of damage to infrastructure, life, and the economy in India. With increasing climate unpredictability, accurate **flood risk prediction** is essential for timely evacuation and planning.  

This project was developed as part of the **Python for Data Science** course, integrating **rainfall, elevation, and flood history data**, performing preprocessing and exploratory analysis, and applying machine learning models to predict flood vulnerability.

---

## ✨ Features
- Data preprocessing: handling missing values, outliers, and merging datasets.
- Exploratory Data Analysis (EDA): histograms, boxplots, correlation heatmaps, scatter plots, and trend analysis.
- Feature engineering: creation of a **Flood Vulnerability Score**.
- Machine Learning Models:
  - Random Forest
  - Linear Regression
  - Logistic Regression
  - Multi-Layer Perceptron (MLP)
- Model evaluation using **MAE, RMSE, F1-score, and Variance Score**.
- ROC curve and AUC analysis for model comparison.

---

## 📊 Datasets
1. **Rainfall Data** – Monthly and seasonal rainfall across Indian subdivisions (1901–2015).  
   Source: [Kaggle](https://www.kaggle.com/datasets/rajanand/rainfall-in-india)  

2. **Elevation Data** – Elevation values (in meters) for Indian districts.  
   Source: [Kaggle](https://www.kaggle.com/datasets/jaisreenivasan/elevation-of-indian-districts)  

3. **Flood History Data** – Flood events across India (1967–2023).  
   Source: [Zenodo](https://zenodo.org/records/11275211)  

4. **Final Merged Dataset** – Combination of rainfall, elevation, and flood history, including engineered vulnerability scores.

---

## 🛠️ Tech Stack
- **Language**: Python  
- **Libraries**: pandas, NumPy, scikit-learn, fuzzywuzzy, matplotlib, seaborn  
- **Environment**: Google Colab / Jupyter Notebook  

---

## ⚙️ Installation & Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/uncodingthecode/Flood-Risk-Prediction.git
   cd Flood-Risk-Prediction
   ```
   
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   If requirements.txt is missing, install manually:
   ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn fuzzywuzzy
   ```
4. Open the notebook:
   ```bash
   jupyter notebook Flood_Risk.ipynb
  or run it directly in Google Colab.

---

## ▶️ Usage
Run all cells in Flood_Risk.ipynb.

The notebook will:
- Load and preprocess datasets.
- Perform EDA (visualizations of rainfall, elevation, flood history).
- Train ML models on merged data.
- Evaluate performance using metrics & ROC curves.
- Output flood risk predictions (High/Medium/Low).

---

## 📈 Results
| Model              | MAE   | F1 Score | RMSE  | Variance Score |
|--------------------|-------|----------|-------|----------------|
| Random Forest      | 0.022 | 0.984    | 0.193 | 0.837          |
| Linear Regression  | 0.263 | 0.987    | 0.382 | 0.979          |
| Logistic Regression| 0.088 | 0.917    | 0.296 | 0.661          |
| MLP                | 0.083 | 0.982    | 0.112 | 0.998          |

✅ Best Performer: Multi-Layer Perceptron (MLP) with highest F1 score and lowest error values.

---

## 👨‍💻 Authors
- Naimish Shah  
- Palash Shah  
- Sk Qeyame Azam  
- Aradya Shetty  
- Aishani Singh  

📌 Guided by **Prof. Nikita Mishra**

---

## 📜 License
This project is for academic purposes. License can be added based on future usage.
