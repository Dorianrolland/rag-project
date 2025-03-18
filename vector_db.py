import os
import concurrent.futures
import hashlib
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from load_documents import load_documents
from langchain.docstore.document import Document

# Répertoire où sera stocké l’index persistant
PERSIST_DIRECTORY = "chroma_db"
DATA_DIRECTORY = "data"

def get_files_hash(directory):
    """
    Génère un hash unique basé sur les fichiers dans un dossier
    pour détecter si des fichiers ont changé.
    """
    hash_md5 = hashlib.md5()
    for root, _, files in os.walk(directory):
        for file in sorted(files):  # Trier pour cohérence
            file_path = os.path.join(root, file)
            with open(file_path, "rb") as f:
                hash_md5.update(f.read())
    return hash_md5.hexdigest()

def chunk_text(text, chunk_size=512, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def chunk_documents(documents, chunk_size=512, overlap=50):
    chunked_docs = []
    for doc in documents:
        text = doc.page_content
        if len(text) > chunk_size:
            chunks = chunk_text(text, chunk_size, overlap)
            for i, chunk in enumerate(chunks):
                new_metadata = doc.metadata.copy()
                new_metadata["chunk_index"] = i
                chunked_docs.append(Document(page_content=chunk, metadata=new_metadata))
        else:
            chunked_docs.append(doc)
    return chunked_docs

def parallel_embed_texts(embeddings, texts):
    """Calcule en parallèle les embeddings pour une liste de textes."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        return list(executor.map(lambda text: embeddings.embed_query(text), texts))

def create_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Vérifier si des fichiers ont changé
    current_hash = get_files_hash(DATA_DIRECTORY)
    hash_file = os.path.join(PERSIST_DIRECTORY, "hash.txt")
    
    if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
        # Vérifier si l'index est à jour
        if os.path.exists(hash_file):
            with open(hash_file, "r") as f:
                saved_hash = f.read().strip()
            if saved_hash == current_hash:
                print("✅ Chargement de l’index existant...")
                return Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        
        print("⚠️ Des modifications détectées, reconstruction de l’index...")

    print("📄 Chargement des documents...")
    documents = load_documents(DATA_DIRECTORY)
    if not documents:
        print("❌ Aucun document trouvé dans 'data/'. Vérifiez vos fichiers !")
        return None
    
    print(f"📂 {len(documents)} documents chargés.")
    
    documents = chunk_documents(documents, chunk_size=512, overlap=50)
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    print("⚡ Calcul des embeddings en parallèle...")
    precomputed_embeddings = parallel_embed_texts(embeddings, texts)

    vector_db = Chroma(embedding_function=embeddings, persist_directory=PERSIST_DIRECTORY)
    vector_db.add_texts(texts, metadatas=metadatas, embeddings=precomputed_embeddings)
    vector_db.persist()

    # Sauvegarder le nouveau hash
    with open(hash_file, "w") as f:
        f.write(current_hash)
    
    print("✅ Base de données vectorielle mise à jour !")
    return vector_db

if __name__ == "__main__":
    vector_db = create_vector_db()
    print("📦 Base de données vectorielle prête !")

