FROM python:3.9-slim

# Installer les dépendances système
RUN apt-get update && apt-get install -y build-essential curl

# Installer Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copier les fichiers de l’application
WORKDIR /app
COPY . /app

# Installer les dépendances Python
RUN pip install --upgrade pip && pip install -r requirements.txt

# Exposer le port du serveur Ollama
EXPOSE 11434
EXPOSE 8000

# Démarrer Ollama, télécharger le modèle Mistral, puis démarrer l’application
CMD ollama serve & sleep 10 && ollama pull mistral && bash

