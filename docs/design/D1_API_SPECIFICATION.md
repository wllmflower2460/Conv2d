# D1.1 API Specification Document
## Conv2d-VQ-HDP-HSMM Behavioral Synchrony Analysis API

**Version**: 1.0.0  
**Status**: Design Gate D1 Preparation  
**Date**: 2025-09-25  
**Authors**: Development Team  

---

## 1. Overview

This document defines the complete API specification for the Conv2d-VQ-HDP-HSMM behavioral synchrony analysis system. The API provides real-time behavioral analysis with uncertainty quantification, supporting both streaming and batch processing modes.

### 1.1 Architecture Reference

```
Client App → EdgeInfer API (8082) → Model Service → Hailo-8/ONNX
                ↓                         ↓              ↓
           Health/Metrics          VQ-HDP-HSMM     Edge Inference
                ↓                         ↓              ↓
           Prometheus              Uncertainty     Performance
                                  Quantification    Optimization
```

### 1.2 Service Endpoints

- **Production**: `https://api.movement.behavioral-sync.io`
- **Staging**: `https://staging-api.movement.behavioral-sync.io`
- **Development**: `http://localhost:8082`
- **Edge Device**: `http://edgepi.local:8082`

---

## 2. Core API Endpoints

### 2.1 Health & Monitoring

#### GET /healthz
**Purpose**: Liveness and readiness probe  
**Authentication**: None required  
**Response Codes**:
- `200 OK` - Service healthy
- `503 Service Unavailable` - Service degraded/unhealthy

**Response Body**:
```json
{
  "status": "healthy",
  "timestamp": "2025-09-25T10:30:45Z",
  "components": {
    "model_service": "healthy",
    "hailo_sidecar": "healthy",
    "database": "healthy"
  },
  "version": "1.0.0",
  "inference_mode": "hailo-8"
}
```

#### GET /metrics
**Purpose**: Prometheus-compatible metrics export  
**Authentication**: Internal only  
**Response**: Plain text Prometheus format

```
# HELP edgeinfer_inference_duration_seconds Inference request duration
# TYPE edgeinfer_inference_duration_seconds histogram
edgeinfer_inference_duration_seconds_bucket{le="0.01"} 234
edgeinfer_inference_duration_seconds_bucket{le="0.025"} 456
edgeinfer_inference_duration_seconds_bucket{le="0.05"} 678
edgeinfer_inference_duration_seconds_bucket{le="0.1"} 891
```

---

## 3. Analysis API (v1)

### 3.1 Session Management

#### POST /api/v1/analysis/start
**Purpose**: Initialize new analysis session  
**Authentication**: Bearer token required  

**Request Headers**:
```
Authorization: Bearer <token>
Content-Type: application/json
X-Correlation-ID: <uuid>
```

**Request Body**:
```json
{
  "subject_id": "subject_123",
  "device_id": "device_456",
  "analysis_type": "behavioral_synchrony",
  "model_config": {
    "vq_mode": "mixed",  // "train", "eval", "mixed"
    "uncertainty_threshold": 0.7,
    "fsq_levels": [5, 4, 4, 3, 3, 3, 2, 2],  // 2,880 codes
    "hdp_enabled": true,
    "hsmm_duration_model": "negative_binomial"
  },
  "metadata": {
    "environment": "clinical",
    "notes": "Initial assessment"
  }
}
```

**Response** (201 Created):
```json
{
  "session_id": "sess_7f8a9b2c3d4e5f6g",
  "created_at": "2025-09-25T10:30:45Z",
  "expires_at": "2025-09-25T11:30:45Z",
  "model_version": "conv2d-vq-hdp-hsmm-v1.0",
  "inference_backend": "hailo-8",
  "calibration": {
    "device": "raspberry-pi-5",
    "expected_latency_ms": 45,
    "max_throughput_hz": 20
  }
}
```

---

### 3.2 Data Streaming

#### PUT /api/v1/analysis/stream
**Purpose**: Stream IMU data for real-time analysis  
**Authentication**: Bearer token required  

**Request Body**:
```json
{
  "session_id": "sess_7f8a9b2c3d4e5f6g",
  "timestamp": 1695638445000,
  "samples": [
    {
      "timestamp_ms": 0,
      "imu": {
        "accel": [0.12, -0.34, 9.81],  // ax, ay, az (m/s²)
        "gyro": [0.01, -0.02, 0.03],   // gx, gy, gz (rad/s)
        "mag": [25.3, -12.1, 48.7]     // mx, my, mz (μT)
      }
    }
    // ... up to 100 samples (1 second window)
  ],
  "window_config": {
    "overlap": 0.5,  // 50% overlap
    "sample_rate": 100  // Hz
  }
}
```

**Response** (202 Accepted):
```json
{
  "status": "accepted",
  "samples_received": 100,
  "buffer_fullness": 0.75,
  "next_analysis_at": "2025-09-25T10:30:46Z"
}
```

---

### 3.3 Behavioral Analysis

#### GET /api/v1/analysis/behavioral
**Purpose**: Get behavioral analysis with uncertainty  
**Authentication**: Bearer token required  

**Query Parameters**:
- `session_id` (required): Active session ID
- `window_size` (optional): Analysis window in ms (default: 1000)
- `include_raw` (optional): Include raw VQ codes (default: false)

**Response** (200 OK):
```json
{
  "session_id": "sess_7f8a9b2c3d4e5f6g",
  "timestamp": "2025-09-25T10:30:45Z",
  "analysis": {
    "behavioral_state": {
      "primary": "synchronized_walking",
      "confidence": 0.85,
      "entropy": 1.23,
      "uncertainty": {
        "aleatoric": 0.15,
        "epistemic": 0.08,
        "total": 0.23
      }
    },
    "vq_codes": {
      "sequence": [12, 45, 12, 78, 45, 12],  // Last 6 codes
      "codebook_utilization": 0.42,
      "perplexity": 127.3
    },
    "hdp_clusters": {
      "active_clusters": 3,
      "assignments": [0, 0, 1, 2, 1, 0],
      "concentration": 1.5
    },
    "hsmm_states": {
      "current_state": 2,
      "duration_remaining": 450,  // ms
      "transition_probability": {
        "to_state_0": 0.1,
        "to_state_1": 0.3,
        "to_state_2": 0.5,
        "to_state_3": 0.1
      }
    },
    "synchrony_metrics": {
      "phase_coupling": 0.73,
      "mutual_information": 2.45,
      "transfer_entropy": {
        "subject_to_companion": 0.34,
        "companion_to_subject": 0.28
      },
      "circular_variance": 0.15
    }
  }
}
```

---

### 3.4 Motif Detection

#### GET /api/v1/analysis/motifs
**Purpose**: Detect behavioral motifs  
**Authentication**: Bearer token required  

**Response** (200 OK):
```json
{
  "session_id": "sess_7f8a9b2c3d4e5f6g",
  "motifs": [
    {
      "id": "motif_001",
      "type": "gait_cycle",
      "score": 0.92,
      "confidence": 0.88,
      "duration_ms": 650,
      "start_time": "2025-09-25T10:30:44.350Z",
      "end_time": "2025-09-25T10:30:45.000Z",
      "characteristics": {
        "periodicity": 0.95,
        "symmetry": 0.87,
        "complexity": 0.43
      }
    }
  ],
  "patterns": {
    "dominant_frequency_hz": 1.8,
    "variability": 0.12,
    "stability_index": 0.89
  }
}
```

---

### 3.5 Stop Analysis

#### POST /api/v1/analysis/stop
**Purpose**: Stop analysis session and get summary  
**Authentication**: Bearer token required  

**Request Body**:
```json
{
  "session_id": "sess_7f8a9b2c3d4e5f6g",
  "save_results": true,
  "export_format": "json"
}
```

**Response** (200 OK):
```json
{
  "session_id": "sess_7f8a9b2c3d4e5f6g",
  "summary": {
    "duration_seconds": 3600,
    "samples_processed": 360000,
    "behavioral_states": {
      "synchronized_walking": 0.45,
      "independent_movement": 0.30,
      "stationary": 0.25
    },
    "quality_metrics": {
      "data_completeness": 0.98,
      "signal_quality": 0.91,
      "analysis_confidence": 0.86
    },
    "export_url": "https://api.movement.behavioral-sync.io/exports/sess_7f8a9b2c3d4e5f6g.json"
  }
}
```

---

## 4. Batch Processing API

### 4.1 Submit Batch Job

#### POST /api/v1/batch/submit
**Purpose**: Submit large dataset for offline processing  
**Authentication**: API key required  

**Request Body**:
```json
{
  "job_name": "clinical_study_001",
  "dataset_url": "s3://datasets/study_001/imu_data.h5",
  "processing_config": {
    "model": "conv2d-vq-hdp-hsmm-v1.0",
    "window_size_ms": 1000,
    "overlap": 0.5,
    "output_format": "hdf5",
    "include_uncertainty": true
  },
  "notification_webhook": "https://callback.example.com/job_complete"
}
```

**Response** (202 Accepted):
```json
{
  "job_id": "job_9f8e7d6c5b4a3",
  "status": "queued",
  "estimated_completion": "2025-09-25T12:30:00Z",
  "tracking_url": "https://api.movement.behavioral-sync.io/batch/status/job_9f8e7d6c5b4a3"
}
```

---

## 5. Model Management API

### 5.1 List Available Models

#### GET /api/v1/models
**Purpose**: List available model versions  
**Authentication**: Bearer token required  

**Response** (200 OK):
```json
{
  "models": [
    {
      "id": "conv2d-vq-hdp-hsmm-v1.0",
      "name": "Production VQ-HDP-HSMM",
      "version": "1.0.0",
      "created_at": "2025-09-20T00:00:00Z",
      "backend": "hailo-8",
      "performance": {
        "latency_p50_ms": 35,
        "latency_p99_ms": 58,
        "throughput_hz": 25,
        "accuracy": 0.812
      },
      "status": "active"
    },
    {
      "id": "fsq-calibrated-v1.1",
      "name": "FSQ Calibrated Model",
      "version": "1.1.0",
      "created_at": "2025-09-24T00:00:00Z",
      "backend": "onnx",
      "status": "beta"
    }
  ]
}
```

---

## 6. Error Responses

### Standard Error Format

All errors follow RFC 7807 Problem Details format:

```json
{
  "type": "https://api.movement.behavioral-sync.io/errors/invalid-session",
  "title": "Invalid Session",
  "status": 404,
  "detail": "Session sess_7f8a9b2c3d4e5f6g not found or expired",
  "instance": "/api/v1/analysis/behavioral?session_id=sess_7f8a9b2c3d4e5f6g",
  "timestamp": "2025-09-25T10:30:45Z",
  "correlation_id": "req_abc123def456"
}
```

### Common Error Codes

| Status | Type | Description |
|--------|------|-------------|
| 400 | `invalid-request` | Malformed request body or parameters |
| 401 | `unauthorized` | Missing or invalid authentication |
| 403 | `forbidden` | Insufficient permissions |
| 404 | `not-found` | Resource not found |
| 409 | `conflict` | Resource state conflict |
| 422 | `validation-error` | Request validation failed |
| 429 | `rate-limited` | Too many requests |
| 500 | `internal-error` | Server error |
| 503 | `service-unavailable` | Service temporarily unavailable |

---

## 7. Authentication & Security

### 7.1 Authentication Methods

1. **Bearer Token** (Primary)
   ```
   Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...
   ```

2. **API Key** (Batch operations)
   ```
   X-API-Key: ak_live_7f8a9b2c3d4e5f6g7h8i9j0k
   ```

### 7.2 Security Headers

Required security headers:
```
X-Correlation-ID: <uuid>
X-Client-Version: 1.0.0
User-Agent: Movement-iOS/1.0.0
```

### 7.3 Rate Limiting

- **Streaming API**: 100 requests/minute per session
- **Analysis API**: 60 requests/minute per user
- **Batch API**: 10 jobs/hour per account

---

## 8. WebSocket API (Real-time)

### 8.1 Connection

```javascript
ws://api.movement.behavioral-sync.io/ws/v1/analysis
```

### 8.2 Message Protocol

**Subscribe to session**:
```json
{
  "type": "subscribe",
  "session_id": "sess_7f8a9b2c3d4e5f6g",
  "auth_token": "Bearer eyJhbGci..."
}
```

**Real-time updates**:
```json
{
  "type": "behavioral_update",
  "session_id": "sess_7f8a9b2c3d4e5f6g",
  "timestamp": "2025-09-25T10:30:45.123Z",
  "data": {
    "state": "synchronized_walking",
    "confidence": 0.87,
    "entropy": 1.15
  }
}
```

---

## 9. Performance Targets

### 9.1 Latency Requirements

| Endpoint | P50 | P95 | P99 |
|----------|-----|-----|-----|
| /healthz | 5ms | 10ms | 20ms |
| /api/v1/analysis/stream | 20ms | 40ms | 60ms |
| /api/v1/analysis/behavioral | 35ms | 65ms | 100ms |
| /api/v1/analysis/motifs | 40ms | 75ms | 120ms |

### 9.2 Throughput Targets

- **Concurrent sessions**: 1000
- **Requests/second**: 500
- **WebSocket connections**: 5000
- **Batch jobs**: 100 concurrent

---

## 10. Deployment Configurations

### 10.1 Edge Device (Raspberry Pi 5 + Hailo-8)

```yaml
deployment:
  platform: raspberry-pi-5
  accelerator: hailo-8
  model_format: hef
  memory_limit: 2GB
  cpu_cores: 4
  inference_threads: 2
```

### 10.2 Cloud (AWS/GCP)

```yaml
deployment:
  platform: kubernetes
  replicas: 3
  resources:
    cpu: 2
    memory: 4Gi
  autoscaling:
    min_replicas: 2
    max_replicas: 10
    target_cpu: 70
```

### 10.3 Mobile (iOS CoreML)

```yaml
deployment:
  platform: ios
  model_format: coreml
  min_ios_version: 15.0
  compute_units: neural_engine
  precision: float16
```

---

## 11. Versioning & Migration

### 11.1 API Versioning

- **URL versioning**: `/api/v1/`, `/api/v2/`
- **Deprecation period**: 6 months
- **Sunset period**: 12 months

### 11.2 Backward Compatibility

- New optional fields can be added
- Existing fields cannot be removed in minor versions
- Breaking changes require major version bump

---

## 12. Testing & Validation

### 12.1 Test Endpoints

Development environment includes test endpoints:

```
POST /api/v1/test/generate_data
GET /api/v1/test/mock_analysis
POST /api/v1/test/simulate_session
```

### 12.2 Sandbox Environment

- **URL**: `https://sandbox.api.movement.behavioral-sync.io`
- **Data**: Synthetic test data only
- **Rate limits**: Relaxed for testing
- **Reset**: Daily at 00:00 UTC

---

## 13. Compliance & Privacy

### 13.1 Data Handling

- **PII**: No personally identifiable information in URLs
- **Encryption**: TLS 1.3 minimum
- **Data retention**: 30 days for active sessions
- **GDPR compliance**: Right to deletion supported

### 13.2 Audit Logging

All API calls logged with:
- Timestamp
- User/Session ID (hashed)
- Endpoint
- Response code
- Latency
- Correlation ID

---

## 14. References

- [EdgeInfer Implementation](../../../pisrv_vapor_docker/README.md)
- [Conv2d-VQ-HDP-HSMM Architecture](../architecture/CONV2D_ARCHITECTURE_DOCUMENTATION.md)
- [Hailo Deployment Guide](../deployment/HAILO_COMPILATION_BREAKTHROUGH.md)
- [Movement Integration](../deployment/MOVEMENT_INTEGRATION_README.md)

---

## Appendix A: Sample Client Implementation

### Python Client Example

```python
import requests
import json
from typing import Dict, List
import numpy as np

class BehavioralSyncClient:
    def __init__(self, api_url: str, auth_token: str):
        self.api_url = api_url
        self.headers = {
            'Authorization': f'Bearer {auth_token}',
            'Content-Type': 'application/json',
            'X-Client-Version': '1.0.0'
        }
        self.session_id = None
    
    def start_session(self, subject_id: str, device_id: str) -> str:
        """Initialize analysis session"""
        response = requests.post(
            f"{self.api_url}/api/v1/analysis/start",
            headers=self.headers,
            json={
                'subject_id': subject_id,
                'device_id': device_id,
                'analysis_type': 'behavioral_synchrony'
            }
        )
        response.raise_for_status()
        data = response.json()
        self.session_id = data['session_id']
        return self.session_id
    
    def stream_imu(self, imu_data: np.ndarray) -> Dict:
        """Stream IMU window (100x9 array)"""
        samples = []
        for i, row in enumerate(imu_data):
            samples.append({
                'timestamp_ms': i * 10,  # 100Hz
                'imu': {
                    'accel': row[0:3].tolist(),
                    'gyro': row[3:6].tolist(),
                    'mag': row[6:9].tolist()
                }
            })
        
        response = requests.put(
            f"{self.api_url}/api/v1/analysis/stream",
            headers=self.headers,
            json={
                'session_id': self.session_id,
                'samples': samples
            }
        )
        response.raise_for_status()
        return response.json()
    
    def get_behavioral_analysis(self) -> Dict:
        """Get current behavioral analysis"""
        response = requests.get(
            f"{self.api_url}/api/v1/analysis/behavioral",
            headers=self.headers,
            params={'session_id': self.session_id}
        )
        response.raise_for_status()
        return response.json()
    
    def stop_session(self) -> Dict:
        """Stop session and get summary"""
        response = requests.post(
            f"{self.api_url}/api/v1/analysis/stop",
            headers=self.headers,
            json={
                'session_id': self.session_id,
                'save_results': True
            }
        )
        response.raise_for_status()
        return response.json()

# Usage example
client = BehavioralSyncClient(
    api_url='http://localhost:8082',
    auth_token='your_token_here'
)

# Start session
session_id = client.start_session('subject_123', 'device_456')
print(f"Started session: {session_id}")

# Stream some data
imu_window = np.random.randn(100, 9) * 0.1  # Simulated IMU data
result = client.stream_imu(imu_window)
print(f"Streamed {result['samples_received']} samples")

# Get analysis
analysis = client.get_behavioral_analysis()
print(f"Behavioral state: {analysis['analysis']['behavioral_state']['primary']}")
print(f"Confidence: {analysis['analysis']['behavioral_state']['confidence']:.2f}")

# Stop session
summary = client.stop_session()
print(f"Session duration: {summary['summary']['duration_seconds']} seconds")
```

---

## Appendix B: OpenAPI Specification

The complete OpenAPI 3.0 specification is available at:
- Development: `http://localhost:8082/openapi.json`
- Production: `https://api.movement.behavioral-sync.io/openapi.json`

---

*End of D1.1 API Specification Document*