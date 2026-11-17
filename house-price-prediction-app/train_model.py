from pathlib import Path
import joblib
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

DATA_CANDIDATES = [
    PROJECT_DIR / "house_price_regression_dataset.csv",
    PROJECT_DIR / "data" / "house_price_regression_dataset.csv",
    APP_DIR / "house_price_regression_dataset.csv",
    APP_DIR / "data" / "house_price_regression_dataset.csv",
]
MODEL_PATH = APP_DIR / "house_price_prediction_model.pkl"

FEATURE_COLUMNS = [
    "Square_Footage",
    "Num_Bedrooms",
    "Num_Bathrooms",
    "Year_Built",
    "Lot_Size",
    "Garage_Size",
    "Neighborhood_Quality",
]
TARGET = "House_Price"

def main():
    data_path = next((p for p in DATA_CANDIDATES if p.is_file()), None)
    if data_path is None:
        raise FileNotFoundError(
            "Dataset not found. Checked:\n" +
            "\n".join(str(p) for p in DATA_CANDIDATES)
        )

    print(f"Dataset: {data_path}")
    df = pd.read_csv(data_path)

    required = FEATURE_COLUMNS + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    if df[required].isna().any().any():
        raise ValueError("Required dataset columns contain missing values.")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET].copy()

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression()),
    ])

    print(f"Training on {len(df)} rows using {len(FEATURE_COLUMNS)} features...")
    pipeline.fit(X, y)

    # Write to a temporary file first, then replace the old blank artifact.
    temporary_path = MODEL_PATH.with_suffix(".tmp")
    joblib.dump(pipeline, temporary_path)
    verified = joblib.load(temporary_path)

    if not hasattr(verified, "predict"):
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError("Saved artifact failed prediction-interface verification.")

    sample_prediction = float(verified.predict(X.iloc[[0]])[0])
    temporary_path.replace(MODEL_PATH)

    print("TRAINING AND SAVE SUCCESSFUL")
    print(f"Model file: {MODEL_PATH}")
    print(f"File size: {MODEL_PATH.stat().st_size:,} bytes")
    print(f"Python package scikit-learn version: {sklearn.__version__}")
    print(f"Sample prediction: {sample_prediction:,.2f}")
    print("The model artifact was loaded again and sample prediction succeeded.")

if __name__ == "__main__":
    main()
