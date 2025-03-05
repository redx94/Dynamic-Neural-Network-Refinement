# API Documentation

The **Dynamic Neural Network Refinement** project includes a FastAPI-based API that serves the trained models for inference. This section provides comprehensive documentation of the API endpoints, including request and response structures, example calls, and authentication mechanisms.

## Overview

The API allows users to perform the following actions:

- **Predict:** Submit input data to receive model predictions.
- **Monitor Metrics:** Access application metrics for monitoring purposes.

## Base URL

All API endpoints are accessible under the base URL:

[http://localhost:8000/](http://localhost:8000/)

## Endpoints

### 1. Predict

**Endpoint:** `/predict`

**Method:** `POST`

**Description:**  
Submits input data to the model and retrieves predictions.

**Request Body:**

```json
{
  "input_data": [[0.1, 0.2, ..., 0.3], [0.4, 0.5, ..., 0.6], ...],
  "current_epoch": 10
}
````

- **input_data:**
    - **Type:** `List[List[float]]`
    - **Description:** A list of input feature vectors. Each inner list represents a single data instance with numerical features.
- **current_epoch:**
    - **Type:** `int`
    - **Description:** The current epoch number, used to adjust model thresholds dynamically.

**Response:**

```json
{
  "predictions": [2, 0, 1, ...]
}
```

- **predictions:**
    - **Type:** `List[int]`
    - **Description:** A list of predicted class labels corresponding to each input data instance.

**Example Request:**

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "input_data": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
           "current_epoch": 10
         }'
```

**Example Response:**

```json
{
  "predictions": [2, 0]
}
```


## Authentication

This API implements API key authentication to secure the `/predict` endpoint. To use the API, you must provide a valid API key in the `X-API-Key` header.

### API Key Setup

1.  **Set the `API_KEY` environment variable:**

    Ensure that the `API_KEY` environment variable is set in your environment. The application will read the API key from this environment variable. For local development, you can set this variable in a `.env` file in the project root directory.

### API Key Usage

To access the `/predict` endpoint, include the `X-API-Key` header in your requests:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: your_secret_api_key" \
     -d '{
           "input_data": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
           "current_epoch": 10
         }'
```

If the `X-API-Key` header is missing or the API key is invalid, the API will return a `403 Forbidden` error.

### Code Implementation

The following code snippet shows how API key authentication is implemented in the `src/app.py` file:

```python
from fastapi import FastAPI, HTTPException, Depends, Header
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_KEY_NAME = "X-API-Key"

async def verify_api_key(api_key_header: str = Header(None, alias=API_KEY_NAME)):
    if api_key_header != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return True

app = FastAPI()

@app.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict(request: InferenceRequest):
    # Your prediction logic here
    return {"predictions": [1, 2, 3]}
```

This example provides a basic implementation of API key authentication. For more advanced security measures, consider using OAuth2 or JWT tokens.

## Error Handling

The API provides meaningful HTTP status codes and error messages to help users understand and resolve issues.

### Common Error Responses

- **400 Bad Request:**
    - **Cause:** Missing or invalid parameters in the request.
        
    - **Response:**
        
        ```json
        {
          "detail": "Invalid input_data format."
        }
        ```
        
- **500 Internal Server Error:**
    - **Cause:** Unexpected errors during processing.
        
    - **Response:**
        
        ```json
        {
          "detail": "An unexpected error occurred. Please try again later."
        }
        ```
        

## Testing the API

You can test the API endpoints using tools like [Postman](https://www.postman.com/) or [cURL](https://curl.se/). Additionally, the API documentation is interactive and can be accessed via the `/docs` endpoint.

**Accessing Interactive API Docs:**

Navigate to `http://localhost:8000/redoc` in your web browser to access the interactive API documentation (Redoc UI).

---

_For more detailed examples and advanced usage, refer to the [Tutorials](docs/tutorials/example_tutorial.md) section._
