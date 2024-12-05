# API Documentation

The **Dynamic Neural Network Refinement** project includes a FastAPI-based API that serves the trained models for inference. This section provides comprehensive documentation of the API endpoints, including request and response structures, example calls, and authentication mechanisms.

## Overview

The API allows users to perform the following actions:

- **Predict:** Submit input data to receive model predictions.
- **Monitor Metrics:** Access application metrics for monitoring purposes.

## Base URL

All API endpoints are accessible under the base URL:

````

[http://localhost:8000/](http://localhost:8000/)

````

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

### 2. Metrics

**Endpoint:** `/metrics`

**Method:** `GET`

**Description:**  
Provides Prometheus-compatible metrics for monitoring application performance and health.

**Response:**

The response will be in the Prometheus text-based exposition format, containing various metrics such as request counts, latency, and resource usage.

**Example Request:**

```bash
curl "http://localhost:8000/metrics"
```

**Example Response:**

```text
# HELP app_requests_total Total number of requests
# TYPE app_requests_total counter
app_requests_total{method="POST",path="/predict"} 150

# HELP app_request_latency_seconds Latency of requests in seconds
# TYPE app_request_latency_seconds histogram
app_request_latency_seconds_bucket{le="0.1"} 50
app_request_latency_seconds_bucket{le="0.5"} 140
app_request_latency_seconds_bucket{le="1.0"} 145
app_request_latency_seconds_bucket{le="+Inf"} 150
app_request_latency_seconds_sum 30.5
app_request_latency_seconds_count 150
```

**Note:**  
These metrics are automatically collected and updated by Prometheus and can be visualized using Grafana.

## Authentication

_Currently, the API does not implement authentication mechanisms. For production environments, it is recommended to secure the API endpoints using authentication methods such as API keys, OAuth2, or JWT tokens._

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

Navigate to `http://localhost:8000/docs` in your web browser to access the Swagger UI, which provides an interactive interface to test the API endpoints.

---

_For more detailed examples and advanced usage, refer to the [Tutorials](https://chatgpt.com/c/tutorials/example_tutorial.md) section._
