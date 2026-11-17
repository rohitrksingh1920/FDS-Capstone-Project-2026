import pandas as pd

FEATURE_COLUMNS = [
    "Square_Footage",
    "Num_Bedrooms",
    "Num_Bathrooms",
    "Year_Built",
    "Lot_Size",
    "Garage_Size",
    "Neighborhood_Quality",
]

def predict_house_price(model, property_details):
    missing = [name for name in FEATURE_COLUMNS if name not in property_details]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    values = [[property_details[name] for name in FEATURE_COLUMNS]]
    frame = pd.DataFrame(values, columns=FEATURE_COLUMNS)
    if frame.isna().any().any():
        raise ValueError("All property features must have numeric values.")
    return float(model.predict(frame)[0])
