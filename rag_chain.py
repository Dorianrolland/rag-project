from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from vector_db import create_vector_db


def create_rag_chain():
    # Créer le retriever
    retriever = create_vector_db().as_retriever(search_kwargs={"k": 10})

    # Charger Ollama avec Llama3
    llm = Ollama(model="llama3")

    # Créer la mémoire de conversation
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # Template pour condenser les questions
    condense_question_prompt = PromptTemplate.from_template("""
    Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:
    """)

    # Chaîne pour générer une nouvelle question claire
    question_generator = LLMChain(
        llm=llm,
        prompt=condense_question_prompt
    )

    # template pour éviter les hallucinations
    qa_template = """
You are a helpful assistant specialized in answering questions based only on the provided context.

Strict rules you must follow:
- Only use the context below to answer the user's question.
- If the context does not contain the information needed, reply: "I don't know based on the provided information."
- NEVER invent any code, method, class, library, or behavior not explicitly mentioned in the context.
- When writing code examples, strictly base them on the provided context.
- Be concise, clear, and stick to factual information.

Context you must use:
{context}

Question: {question}
Answer:"""

    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=qa_template
    )

    # Charger une chaîne de QA
    combine_docs_chain = load_qa_chain(
        llm,
        chain_type="stuff",
        prompt=qa_prompt
    )

    # Créer la chaîne finale
    rag_chain = ConversationalRetrievalChain(
        retriever=retriever,
        memory=memory,
        combine_docs_chain=combine_docs_chain,
        question_generator=question_generator,
        return_source_documents=True,
        output_key="answer"
    )

    return rag_chain

