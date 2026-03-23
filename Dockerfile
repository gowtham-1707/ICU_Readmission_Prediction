FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git curl wget build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/raw data/processed models outputs

CMD ["dvc", "repro"]