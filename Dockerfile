FROM python:3.9-slim

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    tesseract-ocr \
    tesseract-ocr-fra \
    libtesseract-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Installer Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copier les fichiers de l’application
WORKDIR /app
COPY . /app

# Installer les dépendances Python (elles sont toutes listées dans requirements.txt)
RUN pip install --upgrade pip && pip install -r requirements.txt

# (Optionnel) Pour désactiver la parallélisation des tokenizers et éviter des avertissements
ENV TOKENIZERS_PARALLELISM=false

# Exposer les ports
EXPOSE 11434
EXPOSE 8000

# Démarrer Ollama et l’application
CMD ollama serve & sleep 10 && ollama pull mistral && python main.py

