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

# Définir le répertoire de travail
WORKDIR /app

# Copier uniquement les fichiers de configuration et requirements
COPY requirements.txt /app/

# Installer les dépendances Python sans conserver le cache
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Déclarer le volume pour /app pour pouvoir modifier les scripts directement
VOLUME ["/app"]

# Copier les fichiers de l’application (optionnel, sera remplacé par le bind mount)
COPY . /app

# (Optionnel) Pour désactiver la parallélisation des tokenizers et éviter des avertissements
ENV TOKENIZERS_PARALLELISM=false

# Exposer les ports
EXPOSE 11434
EXPOSE 8000

# Démarrer Ollama et l’application
CMD ollama serve & sleep 1 && ollama pull mistral && python main.py
 	
