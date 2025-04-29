import os
import concurrent.futures
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from load_documents import load_documents

# Répertoire où sera stocké l’index persistant
PERSIST_DIRECTORY = "chroma_db"

def parallel_embed_texts(embeddings, texts):
    """
    Calcule en parallèle les embeddings pour une liste de textes.
    Passage à ThreadPoolExecutor pour éviter des problèmes de pickling.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        return list(executor.map(lambda text: embeddings.embed_query(text), texts))

def create_vector_db():
    embeddings = HuggingFaceEmbeddings()
    
    # Vérifier si l'index persistant existe déjà
    if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
        print("Chargement de l’index existant...")
        vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    else:
        print("Création d’un nouvel index...")
        # Charger tous les documents depuis le dossier "data"
        documents = load_documents("data")
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Paralléliser le calcul des embeddings avec ProcessPoolExecutor
        print("Calcul des embeddings en parallèle...")
        precomputed_embeddings = parallel_embed_texts(embeddings, texts)
        
        # Si l'API de Chroma permet d'injecter des embeddings pré-calculés, vous pouvez procéder ainsi :
        vector_db = Chroma(embedding_function=embeddings, persist_directory=PERSIST_DIRECTORY)
        # Supposons que 'add_texts' accepte un paramètre "embeddings" pour des embeddings pré-calculés.
        vector_db.add_texts(texts, metadatas=metadatas, embeddings=precomputed_embeddings)
        
        # Sauvegarde forcée de l'index sur disque
        vector_db.persist()
    return vector_db

if __name__ == "__main__":
    vector_db = create_vector_db()
    print("Base de données vectorielle créée !")

