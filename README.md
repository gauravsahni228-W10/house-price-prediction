# HomeValue AI - House Price Prediction App

A full-stack Machine Learning application predicting residential real estate values using the Kaggle **"House Prices – Advanced Regression Techniques"** dataset.

Built with **Python, FastAPI, scikit-learn, pandas, and joblib**, styled with a clean fintech-inspired responsive UI using **Tailwind CSS & Chart.js**.

---

## Project Structure

```
house-price-prediction/
├── data/
│   └── train.csv               # Kaggle Ames Housing dataset (1,460 samples)
├── models/
│   ├── linear_pipeline.joblib  # Trained Linear Regression pipeline
│   ├── rf_pipeline.joblib      # Trained Random Forest pipeline (200 trees)
│   └── metrics.json            # MAE, RMSE, R², feature importances & test samples
├── static/
│   ├── index.html              # Responsive single-page frontend
│   └── app.js                  # Frontend logic connecting UI to /predict & /metrics
├── train.py                    # Model training, evaluation & serialization script
├── main.py                     # FastAPI backend serving predictions & static files
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation & run guide
```

---

## Quickstart Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Models
```bash
python train.py
```
This script will:
- Load `data/train.csv` (features: `GrLivArea`, `OverallQual`, `BedroomAbvGr`, `FullBath`, `YearBuilt`, `GarageCars`, `TotalBsmtSF`, `Neighborhood`; target: `SalePrice`).
- Impute missing numeric values with median, one-hot encode `Neighborhood`, and scale numeric features for Linear Regression.
- Train both `LinearRegression` and `RandomForestRegressor (n_estimators=200)` on an 80/20 train-test split (`random_state=42`).
- Evaluate **MAE**, **RMSE**, and **R²** on the test set.
- Save both pipelines to `/models` and write evaluation stats to `models/metrics.json`.

### 3. Start the Application Server
```bash
uvicorn main:app --reload
```

Open your browser and navigate to:
**http://127.0.0.1:8000**

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the single-page application |
| `GET` | `/health` | Health check returning `{"status": "ok"}` |
| `GET` | `/metrics` | Returns model evaluation metrics, top features, and test samples |
| `POST` | `/predict` | Predicts price and returns `±8%` range interval |
| `GET` | `/docs` | Interactive Swagger API documentation |

### Example Prediction Request (`POST /predict`):
```json
{
  "living_area": 1710,
  "overall_quality": 7,
  "bedrooms": 3,
  "bathrooms": 2,
  "year_built": 2003,
  "garage_cars": 2,
  "basement_area": 856,
  "neighborhood": "CollgCr",
  "model": "random_forest"
}
```

### Example Response:
```json
{
  "predicted_price": 208750.5,
  "range_low": 192050.46,
  "range_high": 225450.54,
  "model_used": "Random Forest Regressor"
}
```

---

## Model Performance Summary

- **Random Forest Regressor (200 trees)**:
  - $R^2$: **0.8970** (~89.7% variance explained)
  - MAE: **$18,179.73**
  - RMSE: **$28,109.10**
- **Linear Regression (Baseline)**:
  - $R^2$: **0.8303**
  - MAE: **$22,265.27**
  - RMSE: **$36,076.50**

**Top Predictive Features**:
1. Overall Material & Finish Quality (`OverallQual`) - **57.91%**
2. Above Ground Living Area (`GrLivArea`) - **18.38%**
3. Total Basement Area (`TotalBsmtSF`) - **9.94%**
4. Year Built (`YearBuilt`) - **4.89%**
5. Garage Cars Capacity (`GarageCars`) - **3.13%**
