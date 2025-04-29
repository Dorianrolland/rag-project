# Étape 1 : base Python slim
FROM python:3.9-slim

# Étape 2 : Installer dépendances système (pour Tesseract, poppler-utils, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    unzip \
    libgl1 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Étape 3 : Installer Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Étape 4 : Créer le répertoire de travail
WORKDIR /app

# Étape 5 : Copier uniquement requirements.txt
COPY requirements.txt .

# Étape 6 : Installer Python + pip requirements
RUN pip install --upgrade pip && pip install -r requirements.txt

# Étape 7 : Télécharger la ressource nltk 'punkt'
RUN python -m nltk.downloader punkt -d /usr/local/share/nltk_data

# Étape 8 : Copier le reste du projet
COPY . .

# Étape 9 : Définir le répertoire NLTK
ENV NLTK_DATA=/usr/local/share/nltk_data

# Étape 10 : Exposer le port
EXPOSE 8000

# Étape 11 : Commande finale
CMD ollama serve & sleep 2 && ollama pull llama3 && python3 main.py
