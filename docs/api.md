
# API Documentation

## Endpoints

### POST /predict
Makes predictions using the trained model.

**Request**
```json
{
  "input": "image_data_base64"
}
```

**Response**
```json
{
  "prediction": "class_label",
  "confidence": 0.95
}
```

### GET /model/status
Returns model status and metadata.

### POST /model/reload
Reloads model from latest checkpoint.
