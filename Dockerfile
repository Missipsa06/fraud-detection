FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY best_params.json .
COPY app.py .
COPY start.sh .

RUN chmod +x start.sh

# Télécharge les artifacts depuis HF Model Hub pendant le build
RUN python -c "\
from huggingface_hub import hf_hub_download; \
import os; \
os.makedirs('artifacts', exist_ok=True); \
[hf_hub_download('missipsa/fraud-detection-model', f, local_dir='artifacts') \
 for f in ['model.joblib', 'threshold.json', 'samples.json']]"

# Port 7860 = standard Hugging Face Spaces
EXPOSE 7860

CMD ["./start.sh"]
