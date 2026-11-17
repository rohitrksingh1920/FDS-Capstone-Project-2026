# 🏡 House Price Prediction Using Machine Learning

A Machine Learning regression project that predicts house prices using property-related features. This project explores data analysis, preprocessing, feature scaling, regression algorithms, cross-validation, model evaluation, residual analysis, and model persistence using Python and Scikit-learn.

## 📌 Project Overview

House prices depend on several factors, including property size, number of bedrooms, bathrooms, construction year, lot size, garage capacity, and neighborhood quality.

The objective of this project is to develop and evaluate regression models that can accurately predict house prices based on these features.

The project uses a structured dataset containing 1,000 records and implements multiple regression techniques to identify a suitable predictive model.

## 🎯 Objectives

- Understand and explore a house price regression dataset.
- Perform data inspection and exploratory data analysis.
- Prepare input features and target variables.
- Apply feature scaling using StandardScaler.
- Build and compare multiple regression algorithms.
- Evaluate models using appropriate regression metrics.
- Perform cross-validation to assess model consistency.
- Visualize actual vs. predicted values and residuals.
- Save the trained model for future use.

## 📂 Project Structure

```text
House-Price-Prediction/
│
├── Assignment_1_ML_Workshop_Regression.ipynb
│
├── house_price_regression_dataset.csv
│
├── house_price_prediction_model.pkl
│
├── Assignment_1_ML_Workshop_Regression.html
│
├── Assignment_1_ML_Workshop_Regression.pdf
│
└── README.md
```

*Note: HTML, PDF, and model files should be included only if you have generated them and intend to upload them.*

## 📊 Dataset Information

The project uses `house_price_regression_dataset.csv`.

- **Total Records:** 1,000
- **Total Columns:** 8
- **Input Features:** 7
- **Target Variable:** 1

### Dataset Features

| Feature | Description |
|---|---|
| `Square_Footage` | Total area of the house |
| `Num_Bedrooms` | Number of bedrooms |
| `Num_Bathrooms` | Number of bathrooms |
| `Year_Built` | Year the property was constructed |
| `Lot_Size` | Size of the property lot |
| `Garage_Size` | Garage capacity |
| `Neighborhood_Quality` | Neighborhood quality indicator |
| `House_Price` | Target variable to predict |

### Target Variable

`House_Price` is the dependent variable, while the remaining seven columns are used as independent variables.

## ⚙️ Technologies & Libraries

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Jupyter Notebook | Development environment |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine Learning algorithms and preprocessing |
| Joblib | Model persistence |

## 🔬 Project Workflow

### 1. Data Loading and Inspection
- Load the dataset using Pandas.
- Inspect the first and last records.
- Examine dataset dimensions, data types, and structure.
- Generate descriptive statistical summaries.

### 2. Exploratory Data Analysis
- Understand feature distributions.
- Examine relationships between input variables and house prices.
- Use visualizations to understand patterns in the dataset.

### 3. Feature Selection
- Separate independent variables (`X`) and dependent variable (`y`).
- Use `House_Price` as the prediction target.

### 4. Data Splitting and Preprocessing
- Split the dataset into training and testing sets.
- Apply `StandardScaler` to standardize features.
- Fit the scaler on training data and transform testing data using the fitted scaler.

### 5. Regression Model Development

The project explores the following regression algorithms:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net
- Random Forest Regression (cross-validation comparison)

### 6. Model Evaluation

The models are evaluated using:

- **MAE (Mean Absolute Error):** Measures the average absolute prediction error.
- **MSE (Mean Squared Error):** Measures the average squared prediction error.
- **RMSE (Root Mean Squared Error):** Represents prediction error in the target variable's units.
- **R² Score:** Measures how well the model explains the variation in the target variable.

### 7. Cross-Validation

Cross-validation is used to compare model performance across multiple folds and assess consistency.

### 8. Final Model Selection

Linear Regression is selected as the final candidate based on the cross-validation comparison.

A Scikit-learn Pipeline combines:

- StandardScaler
- LinearRegression

### 9. Model Performance Visualization

The project includes:

- Actual vs. Predicted House Prices
- Residual Analysis
- Actual vs. Predicted Price Comparison

### 10. Model Persistence

The trained model is saved using Joblib so that it can be reused without retraining.

## 📈 Model Performance

The final Linear Regression model achieved the following evaluation results on the test dataset:

| Evaluation Metric | Result |
|---|---:|
| MAE | 8,384.71 |
| MSE | 107,941,927.53 |
| RMSE | 10,389.51 |
| R² Score | 0.998338 |

### Cross-Validation Results

| Model | Mean CV R² | Standard Deviation |
|---|---:|---:|
| Linear Regression | 0.998500 | 0.000203 |
| Lasso | 0.998500 | 0.000203 |
| Ridge | 0.998498 | 0.000200 |
| Random Forest | 0.992359 | 0.001286 |

**Observation:** Linear Regression achieved an excellent R² score on the test dataset and performed consistently during cross-validation.

*These results reflect the executed notebook outputs and may vary if the dataset split or model configuration changes.*

## 🚀 Getting Started

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### 2. Navigate to the Project Directory

```bash
cd House-Price-Prediction
```

### 3. Create a Virtual Environment (Recommended)

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter joblib
```

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `Assignment_1_ML_Workshop_Regression.ipynb` and execute the cells.

## 💾 Saved Model

The final trained model is stored as:

```text
house_price_prediction_model.pkl
```

The model can be loaded using Joblib:

```python
import joblib

model = joblib.load("house_price_prediction_model.pkl")
```

The saved Scikit-learn Pipeline includes both feature scaling and the trained regression model.

## 📚 Key Learning Outcomes

Through this project, I gained practical experience in:

- Working with structured datasets.
- Performing exploratory data analysis.
- Preparing data for regression tasks.
- Implementing multiple regression algorithms.
- Understanding feature scaling and pipelines.
- Evaluating models using regression metrics.
- Applying cross-validation.
- Interpreting residual plots.
- Saving and reusing trained Machine Learning models.

## 🔮 Future Improvements

- Develop an interactive house price prediction web application.
- Build a prediction interface using Flask or FastAPI.
- Experiment with additional regression algorithms.
- Perform hyperparameter tuning.
- Improve model generalization using additional datasets.
- Deploy the trained model as an API.

## 👨‍💻 Author

**Rohit Singh**

Artificial Intelligence & Data Science

Interested in Machine Learning, Artificial Intelligence, and building practical AI-powered solutions.

---

⭐ If you find this project useful, consider giving this repository a star!

**Made with Python, Machine Learning, and curiosity.**
