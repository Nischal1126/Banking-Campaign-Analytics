# Bank Marketing Campaign Analytics

A comprehensive data mining project analyzing bank marketing campaign data to predict client subscription to term deposits.

## Overview

This project analyzes the Bank Marketing dataset from UCI Machine Learning Repository to understand factors influencing customer decisions during direct marketing campaigns (phone calls) of a Portuguese banking institution. The goal is to predict whether a client will subscribe to a term deposit.

## Dataset

- **Source**: [UCI Machine Learning Repository - Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- **Period**: May 2008 to November 2010
- **Size**: 41,188 instances with 20 input features

### Features

| Category | Features |
|----------|----------|
| **Client Data** | age, job, marital, education, default, housing, loan |
| **Campaign Data** | contact, month, day_of_week, duration, campaign, pdays, previous, poutcome |
| **Economic Indicators** | emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed |
| **Target** | y (subscribed to term deposit: yes/no) |

## Project Structure

```
Bank_Marketing_Analysis/
├── main.ipynb                 # Main analysis notebook
├── Prep_data/
│   ├── Original_data/         # Raw datasets
│   ├── Combined_data/         # Merged datasets
│   ├── Formated_data/         # Cleaned datasets
│   └── prep_data.ipynb        # Data preparation notebook
├── Report/
│   ├── EDA/
│   │   ├── 00_Univariant_Analysis/
│   │   ├── 10_Bivariant_Analysis/
│   │   └── 20_Multivariant_Analysis/
│   ├── Evaluation/
│   │   ├── confusion_metircs/
│   │   └── ROC_curves/
│   └── XAI/                   # Explainability plots
└── README.md
```

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Univariate analysis with distribution plots and pie charts
- Bivariate analysis examining relationships with target variable
- Multivariate analysis with correlation heatmaps
- Handling class imbalance (only ~11% positive class)

### 2. Data Preprocessing
- Handling missing/unknown values through mode imputation
- Feature engineering:
  - Education level grouping (low, medium, high)
  - Age binning (Young, Adult, Senior, Elderly)
  - Previous contact indicator
- Encoding:
  - StandardScaler for numerical features
  - OneHotEncoder for nominal features
  - OrdinalEncoder for ordinal features

### 3. Machine Learning Models
- **Logistic Regression** (with class balancing)
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**

### 4. Model Evaluation
- Classification reports (Precision, Recall, F1-Score)
- Confusion matrices
- ROC curves with AUC scores

### 5. Model Explainability
- **SHAP** (SHapley Additive exPlanations) for global feature importance
- **LIME** (Local Interpretable Model-agnostic Explanations) for local predictions

## Key Findings

- The dataset is highly imbalanced with only ~11% subscription rate
- Economic indicators (emp.var.rate, euribor3m, nr.employed) strongly correlate with subscription
- Previous campaign success significantly increases future subscription likelihood
- Clients without personal loans are more likely to subscribe
- Campaign timing matters: certain months show higher conversion rates

## Installation

```bash
# Clone the repository
git clone https://github.com/Nischal1126/Banking-Campaign-Analytics.git
cd Banking-Campaign-Analytics

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

```
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
shap
lime
missingno
scipy
```

## Usage

1. Open `main.ipynb` in Jupyter Notebook or VS Code
2. Run all cells sequentially
3. View generated reports in the `Report/` directory

## Model Save & Load

Trained models are saved using `joblib` for reuse:

```python
import joblib
import os

# Save models
model_dir = 'saved_models'
joblib.dump(model, os.path.join(model_dir, 'logistic_regression.pkl'))
joblib.dump(rf_model, os.path.join(model_dir, 'random_forest.pkl'))
joblib.dump(svm, os.path.join(model_dir, 'svm.pkl'))
joblib.dump(preprocessor, os.path.join(model_dir, 'preprocessor.pkl'))
```

```python
# Load models
loaded_lr = joblib.load(os.path.join(model_dir, 'logistic_regression.pkl'))
loaded_rf = joblib.load(os.path.join(model_dir, 'random_forest.pkl'))
loaded_svm = joblib.load(os.path.join(model_dir, 'svm.pkl'))
loaded_preprocessor = joblib.load(os.path.join(model_dir, 'preprocessor.pkl'))

# Make predictions
new_data_processed = loaded_preprocessor.transform(new_data)
predictions = loaded_lr.predict(new_data_processed)
```

## Results

| Model | Accuracy | AUC Score |
|-------|----------|-----------|
| Logistic Regression | ~82% | ~0.78 |
| Random Forest | ~85% | ~0.73 |
| SVM | ~88% | ~0.68 |

*Note: Models use class_weight='balanced' to handle imbalanced data*

## License

This project is for educational purposes.

## Acknowledgments

- UCI Machine Learning Repository for the dataset
- [Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems.
