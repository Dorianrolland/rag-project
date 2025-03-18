import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from PIL import Image
import pytesseract
from pdf2image import convert_from_path  # Assure-toi que pdf2image est installé

def load_pdf(file_path, filename):
    """
    Tente d'extraire le texte du PDF avec PyPDFLoader. 
    Si le texte extrait est très court, on suppose que le PDF est scanné et on applique l'OCR.
    """
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    # Concaténer le texte de toutes les pages
    extracted_text = " ".join([doc.page_content for doc in docs]).strip()
    
    # Si le texte est trop court, on applique l'OCR
    if len(extracted_text) < 50:
        print(f"📄 Le PDF '{filename}' semble être scanné, utilisation de l'OCR via pdf2image.")
        try:
            # Convertir toutes les pages du PDF en images (dpi=300 par exemple)
            pages = convert_from_path(file_path, dpi=300)
            ocr_text = ""
            for page in pages:
                ocr_text += pytesseract.image_to_string(page) + "\n"
            if ocr_text.strip():
                return [Document(page_content=ocr_text, metadata={"source": filename, "type": "pdf-ocr"})]
            else:
                print(f"❌ Aucun texte extrait par OCR pour '{filename}'.")
                return []
        except Exception as e:
            print(f"❌ Erreur lors de l'OCR pour '{filename}': {e}")
            return []
    else:
        return docs

def load_documents(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        lower_name = filename.lower()
        
        if lower_name.endswith(".pdf"):
            try:
                docs = load_pdf(file_path, filename)
                documents.extend(docs)
            except Exception as e:
                print(f"Erreur lors du chargement du PDF '{filename}': {e}")
        
        elif lower_name.endswith((".png", ".jpg", ".jpeg")):
            try:
                image = Image.open(file_path).convert("L")
                text = pytesseract.image_to_string(image)
                documents.append(Document(page_content=text, metadata={"source": filename, "type": "image"}))
            except Exception as e:
                print(f"Erreur lors du traitement de l'image '{filename}': {e}")
        
        elif lower_name.endswith((".py", ".js", ".java", ".c", ".cpp", ".txt")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                documents.append(Document(page_content=text, metadata={"source": filename, "type": "text"}))
            except Exception as e:
                print(f"Erreur lors du chargement du fichier texte '{filename}': {e}")
        
        else:
            print(f"Format non supporté pour le fichier '{filename}'")
    return documents

if __name__ == "__main__":
    docs = load_documents("data")
    print("Documents chargés :", len(docs))

