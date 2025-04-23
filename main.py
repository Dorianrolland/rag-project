from rag_chain import create_rag_chain

def main():
    rag_chain = create_rag_chain()
    while True:
        question = input("Ask your question (or type 'exit' to quit): ")
        if question.lower() == "exit":
            break
        response = rag_chain(question)
        print("Response:", response)

if __name__ == "__main__":
    main()

