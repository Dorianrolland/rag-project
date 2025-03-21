from __future__ import annotations
from typing import List
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from vector_db import create_vector_db
from langchain.schema import BaseRetriever, Document
from sentence_transformers import SentenceTransformer, util
from pydantic import BaseModel

class ReRankingRetriever(BaseRetriever, BaseModel):
    base_retriever: BaseRetriever
    model_name: str = "all-mpnet-base-v2"
    top_k: int = 5  # Limiter à 5 documents pour l'affichage
    _model: SentenceTransformer

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, base_retriever: BaseRetriever, model_name: str = "all-mpnet-base-v2"):
        super().__init__(base_retriever=base_retriever, model_name=model_name)
        self._model = SentenceTransformer(model_name)

    def get_relevant_documents(self, query: str) -> List[Document]:
        # Récupérer tous les documents via le retriever de base
        base_docs = self.base_retriever.get_relevant_documents(query)
        query_embedding = self._model.encode(query, convert_to_tensor=True)
        scored_docs = [
            (doc, util.cos_sim(query_embedding, self._model.encode(doc.page_content, convert_to_tensor=True)).item())
            for doc in base_docs
        ]
        scored_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        # Retourne les 5 premiers documents reclassés
        return [doc for doc, score in scored_docs[:self.top_k]]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Nous avons fourni des informations contextuelles ci-dessous :\n\n"
        "{context}\n\n"
        "Réponds à la question suivante en te basant uniquement sur ces informations.\n"
        "Si aucune information ne permet de répondre, indique-le.\n"
        "Question : {question}"
    )
)

def create_rag_chain():
    llm = OllamaLLM(model="mistral")
    # Demander au retriever de renvoyer jusqu'à 100 documents pour être sûr de couvrir tous les documents
    base_retriever = create_vector_db().as_retriever(search_kwargs={"k": 100})
    retriever = ReRankingRetriever(base_retriever=base_retriever)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template}
    )

    def run_with_context(question):
        retrieved_docs = retriever.get_relevant_documents(question)

        print("\n📜 **Documents utilisés dans le contexte:**")
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"\n🔹 **Document {i} (Source: {doc.metadata.get('source', 'Inconnu')})**")
            # Affiche jusqu'à 500 caractères du contenu
            print(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))

        response = chain.run(question)
        return response

    return run_with_context

