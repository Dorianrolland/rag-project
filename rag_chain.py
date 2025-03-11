from langchain.chains import RetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_ollama import OllamaLLM  # Assurez-vous d'avoir installé et importé le package recommandé
from vector_db import create_vector_db  # Votre fonction pour créer l'index

def create_rag_chain():
    llm = OllamaLLM(model="mistral")
    retriever = create_vector_db().as_retriever()
    # Construction du chain en spécifiant le type (ici "stuff")
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

