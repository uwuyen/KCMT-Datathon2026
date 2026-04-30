KCMT Datathon 2026

## Overview
This repository contains the full solution of **Team KCMT** for **Datathon 2026 – Round 1**.

The competition focuses on transforming raw **e-commerce data** into:
- Actionable business insights  
- A revenue forecasting model  

---

## Project Structure
```
KCMT-Datathon2026/
│
├── Input Data
│ 
├── Part 3 - Sales Forecasting
│ ├── pipeline
│ │  ├── main.py
│ │  ├── predict.py
│ │  ├── preprocess.py
│ │  ├── train.py
│ │  ├── util.py
│ ├── README.md
│ ├── actual_vs_pred.png
│ ├── lgb_importance.png
│ ├── part3_forecast_plot.png
│ ├── part3_forecasting.py
│ ├── part3_solution.ipynb
│ ├── shap_importance.png
│ ├── shap_summary.png
│ └── submission.csv
│
├── Part 1 - MCQ
│ 
├── Part 2 - EDA
│ 
├── README.md
│
└── requirements.txt
``` 
### Part 1 – MCQ (Data Understanding)
- Answer 10 analytical questions  
- Based on dataset exploration  
- Focus on data reasoning & computation  

---

### Part 2 – EDA & Data Visualization
Explore patterns, trends, and relationships in the dataset.

**Analysis Levels:**
- Descriptive – What happened  
- Diagnostic – Why it happened  
- Predictive – What will happen  
- Prescriptive – What should be done  

---

### Part 3 – Sales Forecasting
- Predict **Revenue** for future dates  
- Build **time-series / machine learning models**  

**Evaluation Metrics:**
- MAE (Mean Absolute Error)  
- RMSE (Root Mean Squared Error)  
- R² (Coefficient of Determination)  

---

## Requirements

### Core Libraries
- pandas  
- numpy  
- matplotlib   
- scikit-learn  

### Machine Learning / Forecasting
- xgboost  
- lightgbm  
- statsmodels  

### Visualization & Analysis
- plotly  

### Utilities
- jupyter  
- tqdm  


## Installation

Install all dependencies:

```bash
pip install -r requirements.txt
```


## How to Run
- Clone repository
  + git clone <your-repo-link>
  + cd KCMT-Datathon2026
- Install dependencies
  + pip install -r requirements.txt
- Run notebooks
  + jupyter notebook
