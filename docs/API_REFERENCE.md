# 🌐 API Reference

## Base URL
```
http://localhost:5000
```

---

## Endpoints

### `GET /` — Dashboard
Returns the interactive web dashboard HTML.

---

### `GET /api/status` — Health Check

**Response:**
```json
{
  "status": "online",
  "service": "AI Medical Image Analysis",
  "version": "1.0.0",
  "tasks": ["pneumonia", "skin", "brain"]
}
```

---

### `GET /api/demo` — Run Demo

Returns mock evaluation metrics for the pneumonia detection task.

**Response:**
```json
{
  "task": "pneumonia",
  "metrics": {
    "accuracy": 0.9312,
    "precision": 0.9187,
    "recall": 0.9405,
    "f1": 0.9295,
    "auc": 0.9718
  },
  "confusion_matrix": [[453, 34], [21, 492]],
  "class_names": ["Normal", "Pneumonia"],
  "n_test_samples": 1000,
  "inference_ms": 42
}
```

---

### `POST /api/predict` — Image Prediction

**Request:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `file` | File | JPEG or PNG image |
| `task` | string | `pneumonia` \| `skin` \| `brain` |

**Response:**
```json
{
  "task": "pneumonia",
  "class": "Pneumonia",
  "confidence": 0.8847,
  "gradcam_b64": "iVBORw0KGgo...",
  "model": "MobileNetV2 (Demo)"
}
```

**Error Response (400):**
```json
{ "error": "No file uploaded" }
```

---

### `GET /api/datasets` — Dataset Info

Returns information about all datasets used in the project.

**Response:**
```json
[
  {
    "name": "Chest X-Ray14 (Pneumonia)",
    "size": "5,863 images",
    "source": "Kaggle / NIH",
    "url": "https://www.kaggle.com/...",
    "task": "Binary Classification"
  },
  ...
]
```
