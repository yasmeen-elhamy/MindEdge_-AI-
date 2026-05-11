FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir torch==2.4.0+cpu --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

RUN find /usr -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /root -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

COPY . .

RUN mkdir -p output/rules output/definitions output/graphs \
    output/graph_analysis rag_index chat_logs temp models

EXPOSE 8000

CMD ["uvicorn", "api_service:app", "--host", "0.0.0.0", "--port", "8000"]
