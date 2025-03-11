Le but de ce projet est de construire un RAG ( Retrieval Augmented Generation )

Donc nous voullons associé un LLM local augmenté avec un contexte. Pour qu'il puisse enrichir c'est réponses avec les information qu'oon lui donne vie de la documentation. 
(texte, image, code, ...)

Dans ce projet nous construisons un container Docker dans lequel on place Mistral AI grâce à ollama.

Le processuce de Rag est quant à lui executer par du code python et langchain.

Nous avons créer un sous répertoire dans ce projet nommer "data" qui sert à stocker la documentation que nous voulons mettre en contexte.

Voici les différentes commandes à utilisé pour tester ce projet :

si vous avez le Dockerfile : docker build -t dorianalp38/rag-project:latest .

si non vous pouvez pull l'image dirctement depuis le Docker Hub : docker pull dorianalp38/rag-project:latest

ensuite : docker run -it -p 8000:8000 dorianalp38/rag-project

une fois le container lancé, le téléchargement de LLM mistral AI devrait s'effectuer automatiquement. 

Pour lancer la LLM avec le RAG il suffira alors de run le script python avec : python3 main.py
