RAG Conversationnel avec Mémoire et Récupération de Documents
Description

Ce projet implémente un système de Génération Augmentée par Récupération (RAG) combinant la puissance d'un modèle de langage (LLM) avec une base de données vectorielle pour fournir des réponses pertinentes à des questions posées dans un contexte conversationnel. Le système utilise un modèle Llama3 exécuté localement via Ollama, et des documents externes sont indexés pour enrichir les réponses.
Fonctionnalités :

    Mémoire de conversation : Conservation de l'historique des questions et réponses pour un contexte complet.

    Récupération documentaire : Extraction de passages pertinents à partir d'une base de documents indexée (Chroma).

    Modèle de langage Llama3 : Utilisation du modèle Llama3 via Ollama pour générer des réponses à partir du contexte récupéré.

Installation
Prérequis

Assurez-vous d'avoir les éléments suivants installés :

    Docker : Pour exécuter le projet dans un conteneur.

    Python 3.9+ : Pour exécuter le code du projet.

    Ollama : Utilisé pour charger et exécuter le modèle Llama3.

Étapes d'installation

    Clonez ce dépôt :

git clone https://github.com/ton-utilisateur/ton-repository.git
cd ton-repository

Construisez l'image Docker :

docker build -t rag-project .

Lancez le conteneur Docker :

docker run -it -p 8000:8000 rag-project

Installez les dépendances Python :

pip install -r requirements.txt

Téléchargez le modèle Llama3 via Ollama (si non déjà fait) :

    ollama pull llama3

Configuration

Le projet nécessite un dossier data/ où vous pouvez placer vos documents à indexer. Ces documents seront convertis en vecteurs et stockés dans la base de données vectorielle Chroma. Vous pouvez utiliser des fichiers texte, PDFs, ou tout autre type de contenu pertinent pour votre domaine.
Utilisation
Lancer l'application

Après avoir configuré le projet et installé les dépendances, vous pouvez démarrer l'application :

python main.py

Une fois le serveur lancé, vous pourrez poser des questions via l'interface. Le système utilisera la mémoire et le mécanisme de récupération pour générer des réponses contextualisées basées sur vos documents.
Exemple de dialogue

    Utilisateur : "Qui a inventé le téléphone ?"

    Assistant : "Le téléphone a été inventé par Alexander Graham Bell en 1876. Voici la source : [Document 1]."

Fonctionnement du système

Le système fonctionne en trois étapes :

    Réception de la question : L’utilisateur pose une question.

    Récupération de documents : Les documents pertinents sont extraits de la base de données vectorielle.

    Génération de la réponse : Le modèle Llama3 génère une réponse en utilisant les documents récupérés comme contexte, puis renvoie cette réponse à l'utilisateur.

Structure du projet

Voici une vue d'ensemble de la structure du projet :

/rag-project
├── /data                    # Dossier contenant les documents à indexer
├── /docker                  # Contient les fichiers pour le Docker
│   ├── Dockerfile           # Configuration du Docker
├── /src                     # Code source du projet
│   ├── main.py              # Fichier principal pour exécuter le projet
│   ├── rag_chain.py         # Définition de la chaîne RAG
│   ├── vector_db.py         # Code pour la gestion de la base de données vectorielle
│   ├── load_documents.py    # Script pour charger les documents dans la base
│   └── requirements.txt     # Dépendances Python
└── README.md                # Documentation du projet

Dépendances

Les principales dépendances Python sont listées dans le fichier requirements.txt :

langchain==0.3.0
sentence-transformers==2.2.0
chroma==0.3.0
llama==1.0.0
Ollama==2.0.0

Contributions

Les contributions sont les bienvenues ! Si vous souhaitez améliorer ce projet ou ajouter des fonctionnalités, n'hésitez pas à forker le dépôt et soumettre des demandes de fusion.
Comment contribuer ?

    Forkez ce dépôt.

    Créez une branche pour votre fonctionnalité (git checkout -b feature/ma-fonctionnalite).

    Commitez vos changements (git commit -am 'Ajout d\'une fonctionnalité').

    Poussez la branche (git push origin feature/ma-fonctionnalite).

    Ouvrez une pull request.

Auteurs

    ROLLAND Dorian


