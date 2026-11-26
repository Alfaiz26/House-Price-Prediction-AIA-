# House Price Prediction using Machine Learning (AIA Project)

**Student:** Mohammad Ali Saleem  
**Project:** House Price Prediction  
**Target variable:** `SalePrice`  
**Problem type:** Regression

## Objective
Predict a house's sale price from property features such as area, rooms, age, quality, and location.

## Dataset
The project includes `data/house_prices_sample.csv`, a small **synthetic demonstration dataset** created so the code can run immediately.

For the assignment's Kaggle requirement, download the training file from:
[Kaggle: House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

After downloading, place `train.csv` in the `data/` folder. The program automatically prioritizes `data/train.csv` over the sample dataset. Kaggle's training file contains the `SalePrice` target column.

Dataset reference: Kaggle's competition describes `SalePrice` as the property's sale price and provides the train/test data files. The dataset is the Ames, Iowa housing dataset.

## Workflow
1. Load dataset and inspect rows/columns.
2. Handle missing target values and remove duplicate rows.
3. Impute missing numerical values with the median.
4. Impute missing categorical values with the most frequent category.
5. Encode categorical columns using One-Hot Encoding.
6. Analyze numerical feature correlation.
7. Split data into 80% training and 20% testing.
8. Train Linear Regression and Random Forest Regressor.
9. Evaluate with MAE, RMSE, and R² score.
10. Save charts and result CSV files to `outputs/`.

## Model selection
This is a regression problem because `SalePrice` is a continuous numeric value. Linear Regression provides a simple baseline; Random Forest can model nonlinear relationships and feature interactions. The program compares both and reports the better model on the test split using R² (while also reporting MAE and RMSE).

## Installation and execution
Open a terminal in the project folder and run:

```bash
python -m pip install -r requirements.txt
python main.py
```

## Output files
- `outputs/cleaned_dataset.csv`
- `outputs/model_evaluation.csv`
- `outputs/correlation_heatmap.png`
- `outputs/numerical_correlation.csv`
- `outputs/actual_vs_predicted.png`
- `outputs/actual_vs_predicted_values.csv`
- `outputs/feature_importance.csv`
- `outputs/feature_importance.png`

## Limitations
The included sample is synthetic and is only for demonstration. For the college assignment, use the Kaggle dataset and explain that the training data is from house sales in Ames, Iowa. Predictions are educational and should not be treated as professional real-estate valuations.
