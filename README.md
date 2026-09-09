
## Methodology

1. **Data Preprocessing** — Cleaning, removal of failure-mode indicators (target leakage), encoding categorical variables, feature scaling with StandardScaler, and stratified 80/20 train-test split
2. **Model Training** — Decision Tree, Random Forest, and XGBoost classifiers trained under identical conditions (random_state=42)
3. **Conventional Evaluation** — Accuracy, precision, recall, F1-score, and confusion matrix analysis
4. **False Positive / False Negative Impact Analysis** — Operational cost breakdown showing FN dominates error cost (98–99% of total error cost across all models)
5. **Maintenance Cost Evaluation** — Structured cost matrix assigning £300 per TP (preventive maintenance), £100 per FP (unnecessary inspection), and £8,000 per FN (failure + downtime)
6. **Sensitivity Analysis** — Model ranking comparison under low, medium, and high maintenance cost scenarios
7. **Interpretability** — Feature importance comparison and SHAP value analysis for the XGBoost model
8. **Software Testing** — 10 systematic tests covering functional validation, edge cases, reproducibility, and data integrity

## Interactive Decision-Support Application

The Streamlit application functions as a **maintenance decision-support tool** that connects predicted failure risk with recommended maintenance action, estimated cost, and potential savings. When a failure is predicted, the application recommends scheduling preventive maintenance (£300) and displays the potential failure cost if ignored (£8,000), showing the £7,700 saved by acting on the prediction.

Features:
- Select a trained model (Decision Tree, Random Forest, or XGBoost)
- Input machine sensor values (6 operational features)
- View failure predictions with probability scores
- Compare preventive maintenance cost against potential failure cost
- Inspect feature importance and SHAP explanations for the selected model

### Running the Application

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Software Evaluation

All 10 tests passed:

| # | Test | Result |
|---|------|--------|
| 1 | Model Persistence (Save/Load .pkl) | PASS |
| 2 | Prediction Output Validation | PASS |
| 3 | Confusion Matrix Verification | PASS |
| 4 | Maintenance Cost Function Validation | PASS |
| 5 | Edge Case — Normal Input | PASS |
| 6 | Edge Case — Extreme Operational Input | PASS |
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

## License

This project is submitted for academic purposes. The AI4I 2020 dataset is publicly available on Kaggle.
