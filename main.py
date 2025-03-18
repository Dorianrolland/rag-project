from rag_chain import create_rag_chain

def main():
    rag_chain = create_rag_chain()
    while True:
        question = input("Pose ta question (ou tape 'exit' pour quitter) : ")
        if question.lower() == "exit":
            break
        response = rag_chain(question)
        print("Réponse :", response)

if __name__ == "__main__":
    main()
