from flask import Flask, request, jsonify, send_from_directory
import requests
import os

app = Flask(__name__)

# MLflow server URL
MLFLOW_SERVER = "http://localhost:5000"

@app.route('/')
def index():
    return send_from_directory('page', 'index.html')

@app.route('/invocations', methods=['POST'])
def search():
    # Forward the request to MLflow
    response = requests.post(
        f"{MLFLOW_SERVER}/invocations",
        json=request.json,
        headers={'Content-Type': 'application/json'}
    )
    return jsonify(response.json())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)