from langchain_community.document_loaders import PyPDFLoader


def load_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

if __name__ == "__main__":
    documents = load_documents("data/mon_document.pdf")
    print("Documents chargés :", len(documents))
