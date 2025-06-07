# QuickCart

This project implements a semantic search system for appliances using embeddings generated from a Hugging Face transformer model.

## Project Structure

- `data/` - Contains the appliance data and embeddings
  - `all_appliances.csv` - Original appliance dataset
  - `appliances_embedded.csv` - Generated embeddings for appliances

- `page/` - Contains the web interface
  - `index.html` - Search interface for appliances

- `embed_data.py` - Script to generate embeddings for appliance data
- `deployment.py` - Script to deploy the model to MLflow
- `app.py` - Flask app to serve the web interface

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Generate embeddings (if not already done):
```bash
python embed_data.py
```

3. Deploy the model to MLflow:
```bash
python deployment.py
```

## Running the Application

1. Start the MLflow server:
```bash
mlflow models serve -m "models:/ApplianceSimilarityModel/latest" -p 5000 --no-conda
```

2. In a separate terminal, start the Flask app:
```bash
python app.py
```

3. Access the web interface at:
```
http://localhost:8080
```

## Features

- Semantic search for appliances based on natural language queries
- Price conversion from INR to USD
- Removal of duplicate products
- Display of product images and details
- Links to original product pages

## Technologies Used

- HP AI Studio
- Hugging Face Transformers for embeddings
- MLflow for model deployment 
- App is deployed to Swagger through AIS
- Pandas for data processing
- scikit-learn for similarity calculations
