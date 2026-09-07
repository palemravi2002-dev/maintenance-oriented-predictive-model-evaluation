# Maintenance-Oriented Evaluation Frameworks for Predictive Model Selection in Machine Failure Prediction

A research project submitted in partial fulfilment of the requirements for the degree of **Master of Science** at **Sheffield Hallam University**.

## Overview

Traditional predictive model selection in maintenance relies on statistical metrics such as accuracy, precision, recall, and F1-score, which do not directly reflect maintenance impacts like failure cost, inspection cost, and downtime. This project develops and evaluates a **maintenance-oriented evaluation framework** that integrates conventional predictive performance with structured maintenance cost analysis and model interpretability to support more operationally informed model selection.

## Research Question

> How does maintenance-oriented evaluation based on failure, inspection, and downtime costs change the ranking of machine-failure prediction models compared with conventional predictive metrics?

## Dataset

- **AI4I 2020 Predictive Maintenance Dataset** from [Kaggle](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)
- 10,000 machine observations with 339 failure cases and 9,661 non-failure cases
- Features: Air Temperature, Process Temperature, Rotational Speed, Torque, Tool Wear, and five failure-mode indicators (TWF, HDF, PWF, OSF, RNF)

## Models Evaluated

| Model | Accuracy | Precision | Recall | F1-Score | Maintenance Cost |
|-------|----------|-----------|--------|----------|-----------------|
| Random Forest | 99.90% | 100% | 97.06% | 98.51% | £35,800 |
| XGBoost | 99.90% | 100% | 97.06% | 98.51% | £35,800 |
| Decision Tree | 99.85% | 98.51% | 97.06% | 97.78% | £35,900 |

## Project Structure

```
PredictiveMaintenance/
│
├── Ravi_shaffiled_hallam_software_model.ipynb   # Full ML pipeline (Jupyter Notebook)
├── app.py                                        # Streamlit interactive application
├── requirements.txt                              # Python dependencies
├── README.md                                     # Project documentation
│
└── saved_models/                                 # Trained model files (.pkl)
    ├── decision_tree.pkl
    ├── random_forest.pkl
    ├── xgboost.pkl
    └── scaler.pkl
```

## Methodology

1. **Data Preprocessing** — Cleaning, encoding categorical variables, feature scaling with StandardScaler, and stratified 80/20 train-test split
2. **Model Training** — Decision Tree, Random Forest, and XGBoost classifiers trained under identical conditions
3. **Conventional Evaluation** — Accuracy, precision, recall, F1-score, and confusion matrix analysis
4. **Maintenance Cost Evaluation** — Structured cost matrix assigning £300 per true positive, £100 per false positive, and £8,000 per false negative (failure + downtime)
5. **Sensitivity Analysis** — Model ranking comparison under low, medium, and high maintenance cost scenarios
6. **Interpretability** — Feature importance comparison and SHAP value analysis for the XGBoost model
7. **Software Testing** — 10 systematic tests covering functional validation, edge cases, reproducibility, and data integrity

## Key Findings

- Random Forest and XGBoost achieved the highest predictive performance and the lowest estimated maintenance cost (£35,800)
- Sensitivity analysis confirmed consistent model rankings across all cost scenarios
- SHAP analysis identified Torque, Tool Wear, and Rotational Speed as the strongest continuous predictors, while HDF, PWF, and OSF were the most influential failure-mode indicators
- Feature importance analysis showed HDF as the single most important feature, reaching 0.44 importance in XGBoost

## Interactive Application

The Streamlit application allows users to:
- Select a trained model (Decision Tree, Random Forest, or XGBoost)
- Input machine sensor values and failure-mode signals
- View failure predictions with probability scores
- Compare preventive maintenance cost against potential failure cost
- Inspect feature importance for the selected model

### Running the Application

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Software Evaluation

All 10 tests passed successfully:

| # | Test | Result |
|---|------|--------|
| 1 | Model Persistence (Save/Load .pkl) | PASS |
| 2 | Prediction Output Validation | PASS |
| 3 | Confusion Matrix Verification | PASS |
| 4 | Maintenance Cost Function Validation | PASS |
| 5 | Edge Case — Normal Input | PASS |
| 6 | Edge Case — All Failure Flags | PASS |
| 7 | Reproducibility (random_state=42) | PASS |
| 8 | ROC-AUC Score Evaluation | PASS |
| 9 | Sensitivity Analysis Consistency | PASS |
| 10 | Data Integrity — No Train/Test Leakage | PASS |

## Tech Stack

- **Language:** Python 3
- **ML Libraries:** scikit-learn, XGBoost, SHAP
- **Data Processing:** pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn
- **Application:** Streamlit
- **IDE:** Jupyter Notebook, PyCharm

## Author

**Palem Ravi**
MSc Computing Research Project — Sheffield Hallam University
Supervisor: Jimenez Rodriguez, Alejandro

## License

This project is submitted for academic purposes. The AI4I 2020 dataset is publicly available on Kaggle.
