"""
AIA Project: House Price Prediction using Machine Learning
Dataset target: SalePrice
This project accepts either:
  1) data/train.csv downloaded from Kaggle's House Prices competition, or
  2) data/house_prices_sample.csv (included demo dataset).
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
TARGET = "SalePrice"
RANDOM_STATE = 42

def locate_dataset():
    # Kaggle file takes priority when downloaded and placed in data/.
    for filename in ["train.csv", "house_prices_sample.csv"]:
        path = DATA_DIR / filename
        if path.exists():
            return path
    raise FileNotFoundError(
        "Dataset not found. Place Kaggle train.csv in data/ or use the included "
        "data/house_prices_sample.csv."
    )

def load_and_clean(path):
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found. Available: {list(df.columns)}")

    rows_before = len(df)
    missing_before = int(df.isna().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] > 0].copy()

    # Kaggle Id is an identifier, not a meaningful house characteristic.
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    print("\n--- DATA CLEANING SUMMARY ---")
    print(f"Dataset file: {path}")
    print(f"Rows before cleaning: {rows_before}")
    print(f"Missing values before cleaning: {missing_before}")
    print(f"Duplicate rows removed: {duplicate_count}")
    print(f"Rows after cleaning: {len(df)}")
    print("Remaining feature missing values will be imputed inside the training pipeline.")

    df.to_csv(OUTPUT_DIR / "cleaned_dataset.csv", index=False)
    return df

def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])
    return preprocessor

def evaluate_model(name, model, X_test, y_test):
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2 Score": r2}, pred

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    dataset_path = locate_dataset()
    df = load_and_clean(dataset_path)

    # Correlation analysis for numerical columns.
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] >= 2:
        plt.figure(figsize=(10, 7))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap - House Price Features")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=300)
        plt.close()
        numeric_df.corr().to_csv(OUTPUT_DIR / "numerical_correlation.csv")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    print(f"\nTraining records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, min_samples_leaf=2, n_jobs=-1
        )
    }

    results, predictions = [], {}
    fitted_models = {}
    for name, estimator in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessing", build_preprocessor(X_train)),
            ("model", estimator)
        ])
        pipeline.fit(X_train, y_train)
        result, pred = evaluate_model(name, pipeline, X_test, y_test)
        results.append(result)
        predictions[name] = pred
        fitted_models[name] = pipeline

    results_df = pd.DataFrame(results).sort_values("R2 Score", ascending=False)
    results_df.to_csv(OUTPUT_DIR / "model_evaluation.csv", index=False)
    print("\n--- MODEL EVALUATION ---")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    best_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    best_pred = predictions[best_name]

    comparison = pd.DataFrame({
        "Actual_SalePrice": y_test.to_numpy(),
        "Predicted_SalePrice": best_pred
    })
    comparison.to_csv(OUTPUT_DIR / "actual_vs_predicted_values.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, best_pred, alpha=0.65)
    min_value = min(float(y_test.min()), float(best_pred.min()))
    max_value = max(float(y_test.max()), float(best_pred.max()))
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")
    plt.xlabel("Actual Sale Price")
    plt.ylabel("Predicted Sale Price")
    plt.title(f"Actual vs Predicted Sale Price ({best_name})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=300)
    plt.close()

    # Feature importance is available for the Random Forest model.
    rf_pipeline = fitted_models["Random Forest Regressor"]
    rf_model = rf_pipeline.named_steps["model"]
    preprocessor = rf_pipeline.named_steps["preprocessing"]
    try:
        feature_names = preprocessor.get_feature_names_out()
        importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": rf_model.feature_importances_
        }).sort_values("Importance", ascending=False)
        importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
        top = importance.head(15).sort_values("Importance")
        plt.figure(figsize=(10, 7))
        plt.barh(top["Feature"], top["Importance"])
        plt.xlabel("Importance")
        plt.title("Top 15 Feature Importances (Random Forest)")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=300)
        plt.close()
    except Exception as exc:
        print(f"Feature importance chart could not be generated: {exc}")

    print(f"\nBest model on this test split: {best_name}")
    print("Outputs saved in the outputs/ folder.")
    print("Note: This model demonstrates associations in the dataset; it is not a professional property valuation.")

if __name__ == "__main__":
    main()
