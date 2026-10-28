"""
House Price Prediction - Model Training Script
College Project: Machine Learning with scikit-learn
Dataset: Kaggle House Prices – Advanced Regression Techniques (train.csv)
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Define file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
LINEAR_MODEL_PATH = os.path.join(MODELS_DIR, "linear_pipeline.joblib")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "rf_pipeline.joblib")

# Selected features according to specification
NUMERIC_FEATURES = [
    "GrLivArea",      # Above grade (ground) living area square feet
    "OverallQual",    # Overall material and finish quality (1-10)
    "BedroomAbvGr",   # Bedrooms above grade
    "FullBath",       # Full bathrooms above grade
    "YearBuilt",      # Original construction date
    "GarageCars",     # Size of garage in car capacity
    "TotalBsmtSF"     # Total square feet of basement area
]

CATEGORICAL_FEATURES = [
    "Neighborhood"    # Physical locations within Ames city limits
]

TARGET_COLUMN = "SalePrice"


def load_and_prepare_data(file_path: str):
    """
    Load dataset from CSV and select only the required features and target.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found at {file_path}. Please place 'train.csv' in the 'data' folder."
        )

    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)

    # Validate that required columns exist in the dataset
    all_needed_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
    missing_cols = [col for col in all_needed_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")

    # Subset the dataframe to only use the specified columns
    selected_df = df[all_needed_columns].copy()

    X = selected_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = selected_df[TARGET_COLUMN]

    print(f"Dataset loaded successfully: {X.shape[0]} samples, {X.shape[1]} features.")
    return X, y


def build_preprocessors():
    """
    Create preprocessing pipelines using scikit-learn ColumnTransformer.
    - Fills missing numeric values with median.
    - One-hot encodes Neighborhood (handles unknown categories gracefully).
    - Scales numeric features for Linear Regression.
    """
    # Preprocessor for Linear Regression (includes StandardScaler)
    numeric_transformer_scaled = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Preprocessor for Random Forest (tree-based models don't require feature scaling)
    numeric_transformer_unscaled = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor_linear = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer_scaled, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )

    preprocessor_rf = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer_unscaled, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )

    return preprocessor_linear, preprocessor_rf


def clean_feature_name(name: str) -> str:
    """Format scikit-learn one-hot and numeric feature names cleanly."""
    if name.startswith("num__"):
        return name.replace("num__", "")
    if name.startswith("cat__Neighborhood_"):
        return f"Neighborhood ({name.replace('cat__Neighborhood_', '')})"
    return name


def main():
    # Ensure models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load data
    X, y = load_and_prepare_data(DATA_PATH)

    # 2. 80/20 train-test split with fixed random_state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Train set size: {len(X_train)} | Test set size: {len(X_test)}")

    # 3. Build preprocessing components
    preprocessor_linear, preprocessor_rf = build_preprocessors()

    # 4. Construct complete Scikit-Learn Pipelines
    linear_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor_linear),
        ("regressor", LinearRegression())
    ])

    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor_rf),
        ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
    ])

    # 5. Train Linear Regression Model
    print("Training Linear Regression model...")
    linear_pipeline.fit(X_train, y_train)
    y_pred_linear = linear_pipeline.predict(X_test)

    # 6. Train Random Forest Regressor Model
    print("Training Random Forest model (n_estimators=200)...")
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)

    # 7. Evaluate both models on test set (MAE, RMSE, R²)
    # Using np.sqrt(mean_squared_error(...)) for wide scikit-learn version compatibility
    mae_linear = mean_absolute_error(y_test, y_pred_linear)
    rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))
    r2_linear = r2_score(y_test, y_pred_linear)

    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)

    print("\n--- MODEL EVALUATION RESULTS ---")
    print(f"Linear Regression -> MAE: ${mae_linear:,.2f} | RMSE: ${rmse_linear:,.2f} | R²: {r2_linear:.4f}")
    print(f"Random Forest     -> MAE: ${mae_rf:,.2f} | RMSE: ${rmse_rf:,.2f} | R²: {r2_rf:.4f}")

    # 8. Extract top 5 Random Forest feature importances
    rf_model = rf_pipeline.named_steps["regressor"]
    rf_prep = rf_pipeline.named_steps["preprocessor"]
    raw_feature_names = rf_prep.get_feature_names_out()
    importances = rf_model.feature_importances_

    sorted_indices = np.argsort(importances)[::-1][:5]
    top_5_features = [
        {
            "feature": clean_feature_name(raw_feature_names[i]),
            "importance": round(float(importances[i]), 4),
            "percentage": round(float(importances[i]) * 100, 2)
        }
        for i in sorted_indices
    ]

    print("\nTop 5 Feature Importances (Random Forest):")
    for idx, item in enumerate(top_5_features, 1):
        print(f"  {idx}. {item['feature']}: {item['percentage']}%")

    # 9. Create a sample of actual-vs-predicted test values (~100 points)
    sample_count = min(100, len(y_test))
    # Pick 100 evenly spaced points from test set
    sample_indices = np.linspace(0, len(y_test) - 1, sample_count, dtype=int)
    y_test_array = y_test.values

    actual_vs_predicted = [
        {
            "id": int(i + 1),
            "actual": round(float(y_test_array[idx]), 2),
            "predicted_linear": round(float(y_pred_linear[idx]), 2),
            "predicted_rf": round(float(y_pred_rf[idx]), 2)
        }
        for i, idx in enumerate(sample_indices)
    ]

    # 10. Save trained pipelines using joblib
    print(f"\nSaving Linear Regression pipeline to {LINEAR_MODEL_PATH}...")
    joblib.dump(linear_pipeline, LINEAR_MODEL_PATH)

    print(f"Saving Random Forest pipeline to {RF_MODEL_PATH}...")
    joblib.dump(rf_pipeline, RF_MODEL_PATH)

    # 11. Compile and save metrics.json
    metrics_data = {
        "models": {
            "linear": {
                "name": "Linear Regression",
                "mae": round(float(mae_linear), 2),
                "rmse": round(float(rmse_linear), 2),
                "r2": round(float(r2_linear), 4)
            },
            "random_forest": {
                "name": "Random Forest Regressor",
                "mae": round(float(mae_rf), 2),
                "rmse": round(float(rmse_rf), 2),
                "r2": round(float(r2_rf), 4)
            }
        },
        "top_features": top_5_features,
        "actual_vs_predicted": actual_vs_predicted,
        "metadata": {
            "total_samples": int(len(X)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "train_ratio": 0.80,
            "test_ratio": 0.20,
            "random_state": 42,
            "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
            "target": TARGET_COLUMN
        }
    }

    print(f"Saving metrics and evaluation data to {METRICS_PATH}...")
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    print("Training process finished successfully!")


if __name__ == "__main__":
    main()
