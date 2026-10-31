# Multimodal Deep Learning Framework for Heart Disease Prediction
### Combining 12-Lead Electrocardiogram (ECG) Signals and Structured Clinical Tabular Data

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.3+](https://img.shields.io/badge/PyTorch-2.3+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-21%20Passed-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/Status-Research%20Prototype%20(RUO)-amber.svg)]()

> **CRITICAL CLINICAL & REGULATORY DISCLAIMER:**
> **Investigational & Research / Educational Use Only (RUO / IUO).**
> This software is not a cleared medical device and must never be used for autonomous patient diagnosis, triage, screening, or clinical decision-making. Model outputs are experimental probability scores and attributions, not clinical diagnoses or medical advice.

---

## 1. Overview & Architectural Highlights

This repository provides an open, mathematically rigorous, and leakage-safe deep-learning framework for multimodal cardiovascular risk prediction. It addresses the siloed nature of traditional models by learning joint representations from:
- **Electrophysiological Data:** Raw/filtered 12-lead ECG time-series ($500\text{ Hz}$, $5.0\text{ seconds}$ = 2,500 samples per lead).
- **Biometric & Biomarker Data:** Tabular clinical features (age, sex, resting blood pressure, cholesterol, fasting glucose, resting ECG abnormalities, maximum heart rate, exercise angina, ST depression, ST slope, fluoroscopy vessels, thalassemia status).

```text
               12-Lead ECG [Batch, 12, 2500]
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   Multi-Scale 1D CNN                 SciPy Bandpass & Notch
 (Kernels 3, 7, 15 + Res)             (0.5-45 Hz, 50/60 Hz)
            │                                 │
            ▼                                 │
     BiLSTM Recurrent                         │
     Temporal Encoder                         │
            │                                 │
            ▼                                 │
    Temporal Attention                        │
            │                                 │
     ECG Embedding (z_ecg)                    │
            │                                 │
            └──────────────┬──────────────────┘
                           │
                           ▼
                  DYNAMIC GATED FUSION  ◄──────  Clinical Embedding (z_clin)
             g = σ(W_g [z_ecg; z_clin] + b_g)                 ▲
                           │                                  │
                           ▼                          3-Layer MLP Encoder
                Classification MLP Head                       ▲
                           │                                  │
                           ▼                       Missingness-Aware Imputer
                  Research Risk Score                      & Scaler
                    (Probability)                             ▲
                           │                                  │
                           ▼                         Tabular Clinical Dict
             Explainability: Saliency &                 (12 Features)
                 Feature Attributions
```

### Key Engineering Capabilities
1. **Multi-Scale Convolutional Pathway:** Parallel 1D CNN receptive fields capture acute high-frequency QRS complexes and low-frequency ST/T-wave morphology simultaneously.
2. **BiLSTM Temporal Dynamics:** Bidirectional LSTM models rhythm variability across heartbeats.
3. **Dynamic Gated Modality Fusion:** Learns adaptive modality weights $g \in [0, 1]$ to dynamically modulate reliance on electrophysiology vs clinical risk factors.
4. **Leakage-Safe Patient-Level Splits:** Strict `GroupShuffleSplit` on `patient_id` guarantees zero patient overlap across training, validation, and test sets.
5. **Explainability Engine:** Integrated Gradients identify salient ECG lead segments (e.g. ST segment elevation/depression), paired with gradient-based clinical feature attributions.
6. **Strict CSV Data Ingestion:** All cohort datasets are stored, parsed, and loaded strictly as CSV (`.csv`) files; non-CSV data files are removed.
7. **Production FastAPI Service:** Schema validation (Pydantic v2), OAuth2/JWT authentication, RBAC, database audit telemetry, and a built-in modern dark-mode Research Dashboard.

---

## 2. Directory Structure

```text
heart_disease_multimodal_framework/
├── PRD.md                             # Comprehensive Product Requirements Document
├── README.md                          # Platform documentation
├── requirements.txt                   # Production Python dependencies
├── .env.example                       # Configuration template
├── .env                               # Active local configuration
├── app.py                             # FastAPI service and dashboard entry point
├── train.py                           # CLI training pipeline (strictly ingests CSV)
├── evaluate.py                        # CLI test evaluation & calibration reporter (CSV ingestion)
├── data/                              # Dataset repository (STRICTLY CSV ONLY)
│   ├── heart_disease_multimodal.csv   # Unified multimodal CSV (clinical features + 12 ECG leads)
│   └── heart_disease.csv              # Canonical clinical dataset (.csv)
├── src/
│   ├── config.py                      # Pydantic Settings configuration
│   ├── api/                           # FastAPI routers, schemas, dependencies
│   ├── data/                          # CSV loader, signal processor, clinical imputer, dataset
│   │   ├── csv_loader.py              # Ingestion, validation, schemas, and CSV parsing
│   │   ├── ecg_processor.py           # Butterworth 0.5-45 Hz filtering & normalization
│   │   ├── clinical_processor.py      # Missingness-aware imputer and standard scaler
│   │   └── dataset.py                 # PyTorch MultimodalDataset & patient-level split
│   ├── models/                        # CNN-BiLSTM, MLP, Fusion, Explainability, Artifact manager
│   ├── database/                      # SQLAlchemy models, sessions, and repository service
│   └── security/                      # Password hashing (bcrypt), JWT tokens, RBAC
├── static/                            # Interactive Web Dashboard (Canvas ECG, Sandbox, UI)
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
├── notebooks/                         # Complete step-by-step reproducible research notebook
│   └── multimodal_heart_disease_pipeline.ipynb
├── tests/                             # Unit and integration test suite (21 test cases)
│   ├── test_data_processing.py        # Includes CSV validation and non-CSV rejection tests
│   ├── test_models.py
│   └── test_api.py
└── artifacts/                         # Serialized model checkpoints (.pt) and model cards
    └── best_multimodal_model.pt
```

---

## 3. Quick Start

### 3.1 Setup Environment

```bash
# 1. Create and activate virtual environment
virtualenv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Initialize environment variables
cp .env.example .env
```

### 3.2 Run Automated Tests
```bash
pytest -v
```

### 3.3 Train and Benchmark Models
Train unimodal baselines and the multimodal gated fusion model on a patient-grouped cohort loaded strictly from CSV:
```bash
python train.py --data data/heart_disease_multimodal.csv --epochs 15 --fusion gated
```
This produces:
- Locked holdout test set evaluation table (comparing ECG-Only vs Clinical-Only vs Multimodal)
- Optimal decision threshold calculation using Youden's J statistic
- Serialized, integrity-hashed model artifact at `artifacts/best_multimodal_model.pt`

### 3.4 Independent Evaluation on CSV Cohort
Evaluate a saved artifact against an independent test set stored in a CSV file:
```bash
python evaluate.py --artifact artifacts/best_multimodal_model.pt --data data/heart_disease_multimodal.csv
```

### 3.5 CSV Data Specification & Storage
All datasets used by the framework are stored strictly in **CSV format (`.csv`)**. Any legacy or non-CSV data files (e.g., SQLite `.db` databases) have been removed.
- **`data/heart_disease_multimodal.csv`**: Complete multimodal dataset containing clinical variables and all 12 ECG lead time-series signals (`lead_I` through `lead_V6`).
- **`data/heart_disease.csv`**: Canonical tabular heart disease records with 12 physiological features, `record_id`, `patient_id`, and binary disease `target`.

### 3.6 Launch FastAPI Web Application & Dashboard
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive Research Sandbox:** Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Swagger Interactive API Documentation:** Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **OpenAPI JSON Schema:** Open [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)

---

## 4. REST API Endpoints

### Default Credentials
| Email | Password | Role | Permissions |
|---|---|---|---|
| `admin@research.cardio.ai` | `AdminSecure2026!` | `admin` | Full control, audit logs, model swapping |
| `researcher@cardio.ai` | `Research2026!` | `researcher` | Predict, explain, view experiments |
| `api_client@cardio.ai` | `ApiClient2026!` | `api_client` | Inference only (`/v1/predict`) |

### Example cURL Request

```bash
# 1. Obtain JWT Bearer Token
TOKEN=$(curl -s -X POST "http://127.0.0.1:8000/v1/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=researcher@cardio.ai&password=Research2026!" | jq -r '.access_token')

# 2. Run Multimodal Prediction
curl -X POST "http://127.0.0.1:8000/v1/predict" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "record_id": "REC_90214",
    "sampling_rate_hz": 500.0,
    "ecg": [[0.012, 0.045, -0.015, ... 2500 samples per lead ...]],
    "clinical": {
      "age": 64.0,
      "sex": 1.0,
      "resting_bp": 155.0,
      "cholesterol": 268.0,
      "fasting_bs": 1.0,
      "resting_ecg": 2.0,
      "max_hr": 128.0,
      "exercise_angina": 1.0,
      "st_depression": 2.4,
      "st_slope": 1.0,
      "num_major_vessels": 2.0,
      "thal": 2.0
    }
  }'
```

### Example Response
```json
{
  "request_id": "8fa8d145-22d7-4f91-953e-52b12361ef56",
  "model_version": "v1.0.0-gated",
  "research_score": 0.8142,
  "decision_threshold": 0.512,
  "prediction_label": "Elevated Risk Indication (Experimental)",
  "modality_contributions": {
    "ecg_weight": 0.584,
    "clinical_weight": 0.416
  },
  "latency_ms": 24,
  "disclaimer": "Research prototype only. Outputs are experimental model scores, not medical diagnoses or clinical advice."
}
```

---

## 5. Security & Privacy Governance

1. **HIPAA Safe Harbor & DPDP Compliance:**
   - No direct identifiers (PHI/PII) accepted or stored in relational tables.
   - Pseudonymous `record_id` and `patient_id` hashing with SHA-256 salts.
2. **Fail-Closed API Behavior:**
   - If an approved model artifact is missing or corrupted, the service returns `503 Service Unavailable` rather than returning fabricated or random scores.
3. **Audit Telemetry:**
   - Every inference event logs execution latency, actor, outcome status, and unique `request_id` without persisting raw biomedical waveforms in access logs.

---

## 6. License & Citation
Licensed under the Apache 2.0 License. For research and educational usage only.
