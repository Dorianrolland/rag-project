from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from load_documents import load_documents

def create_vector_db():
    documents = load_documents("data/mon_document.pdf")
    embeddings = HuggingFaceEmbeddings()
    vector_db = Chroma.from_documents(documents, embeddings)
    return vector_db

if __name__ == "__main__":
    vector_db = create_vector_db()
    print("Base de données vectorielle créée !")
