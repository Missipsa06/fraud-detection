#!/bin/bash
set -e

# Lance FastAPI en arrière-plan (port 8000, interne)
uvicorn src.api:app --host 0.0.0.0 --port 8000 &

# Lance Streamlit en avant-plan (port 7860, exposé par HF Spaces)
streamlit run app.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.headless true
