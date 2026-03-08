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
RUN huggingface-cli download missipsa/fraud-detection-model \
    --local-dir artifacts/ \
    --repo-type model

# Port 7860 = standard Hugging Face Spaces
EXPOSE 7860

CMD ["./start.sh"]
