# HomeValue AI — Final Setup

## 1. Required project files

Keep `house_price_regression_dataset.csv` in the parent project folder, or in this app folder's `data/` directory.

The trained model is generated locally by `train_model.py`. Do not create or edit the `.pkl` file manually.

## 2. Train the model (Windows)

Open CMD/VS Code terminal in this folder:

```cmd
python -m pip install -r requirements.txt
python train_model.py
```

The script trains the saved StandardScaler + LinearRegression pipeline, writes `house_price_prediction_model.pkl`, reloads it, and verifies a sample prediction.

## 3. Run locally

```cmd
python -m streamlit run app.py
```

Open http://localhost:8501.

## 4. Docker

Run these commands from this folder, after training the model:

```cmd
docker build -t homevalue-ai .
docker run --rm -p 8501:8501 --name homevalue-ai homevalue-ai
```

Open http://localhost:8501.

## Important model note

Joblib files are Python/library-version-sensitive. This project pins the versions shown in the user's reported Python 3.11 environment. Train the model in the same environment used to run the app/container. Do not copy a zero-byte placeholder model into Docker.

The evaluation metrics and cross-validation values shown in the dashboard are those recorded in the original notebook; the deployment training script does not recalculate or replace those evaluation results.


## Visual theme
The interface uses an Obsidian + Emerald palette: Obsidian `#101714`, deep surface `#18221D`, Emerald `#2FBF85`, and soft white `#F2F6F3`.
