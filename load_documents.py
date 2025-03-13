import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document

from PIL import Image
import pytesseract

def load_documents(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        lower_name = filename.lower()
        
        # Charger les PDF
        if lower_name.endswith(".pdf"):
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                print(f"Erreur lors du chargement du PDF {filename}: {e}")
        
        # Charger les images (.png, .jpg, .jpeg) avec OCR
        elif lower_name.endswith((".png", ".jpg", ".jpeg")):
            try:
                image = Image.open(file_path).convert("L")
                text = pytesseract.image_to_string(image)
                documents.append(Document(page_content=text, metadata={"source": filename, "type": "image"}))
            except Exception as e:
                print(f"Erreur lors du traitement de l'image {filename}: {e}")
        
        # Charger les fichiers de code ou texte (.py, .js, .java, .c, .cpp, .txt)
        elif lower_name.endswith((".py", ".js", ".java", ".c", ".cpp", ".txt")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                documents.append(Document(page_content=text, metadata={"source": filename, "type": "text"}))
            except Exception as e:
                print(f"Erreur lors du chargement du fichier texte {filename}: {e}")
        
        else:
            print(f"Format non supporté pour le fichier {filename}")
    return documents

if __name__ == "__main__":
    docs = load_documents("data")
    print("Documents chargés :", len(docs))

