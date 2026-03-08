#!/bin/bash
set -e

# Télécharge les artifacts depuis HF Model Hub si absents
python - <<'EOF'
from huggingface_hub import hf_hub_download
import os

os.makedirs("artifacts", exist_ok=True)
for filename in ["model.joblib", "threshold.json", "samples.json"]:
    if not os.path.exists(f"artifacts/{filename}"):
        print(f"Downloading {filename}...")
        hf_hub_download(
            repo_id="missipsa/fraud-detection-model",
            filename=filename,
            local_dir="artifacts",
        )
print("Artifacts ready.")
EOF

# Lance FastAPI en arrière-plan (port 8000, interne)
uvicorn src.api:app --host 0.0.0.0 --port 8000 &

# Lance Streamlit en avant-plan (port 7860, exposé par HF Spaces)
streamlit run app.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.headless true
