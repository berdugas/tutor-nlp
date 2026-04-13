# AralMate NLP Service

Tagalog linguistic analysis microservice using calamanCy.

## Endpoints
- GET  /health         — health check
- POST /filter-words   — filter word list by POS/NER
- POST /analyze        — full text analysis

## Auth
All POST endpoints require header: X-AralMate-NLP-Secret: <secret>

## Local development
pip install -r requirements.txt
uvicorn main:app --reload
