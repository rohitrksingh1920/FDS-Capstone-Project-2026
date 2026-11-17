
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install application dependencies
COPY house-price-prediction-app/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the Streamlit application and its supporting modules
COPY house-price-prediction-app/ ./house-price-prediction-app/

# Copy the original dataset into the project root.
# app.py expects the dataset one directory above its application folder.
COPY house_price_regression_dataset.csv ./house_price_regression_dataset.csv

EXPOSE 8501

CMD ["streamlit", "run", "house-price-prediction-app/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
