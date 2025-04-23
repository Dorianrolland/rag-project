FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    tesseract-ocr \
    tesseract-ocr-fra \
    libtesseract-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Copy only configuration files and requirements
COPY requirements.txt /app/

# Install Python dependencies without caching
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Declare volume for /app to modify scripts directly
VOLUME ["/app"]

# Copy application files (optional, will be replaced by bind mount)
COPY . /app

# Disable tokenizer parallelism to avoid warnings
ENV TOKENIZERS_PARALLELISM=false

# Expose ports
EXPOSE 11434
EXPOSE 8000

# Start Ollama and the application
CMD ollama serve & sleep 1 && ollama pull mistral && python main.py

