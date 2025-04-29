from rag_chain import create_rag_chain

def main():
    rag_chain = create_rag_chain()

    print("🔍 Ready to chat! Ask your questions.\n")

    while True:
        query = input("🗨️  Your question (type 'exit' to quit): ")

        if query.lower() == 'exit':
            break

        result = rag_chain.invoke({
            "question": query
        })

        print(f"\n🤖 Answer:\n{result['answer']}\n")
        print("-" * 50)


if __name__ == "__main__":
    main()

