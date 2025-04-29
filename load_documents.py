import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

def load_pdf(file_path, filename):
    """
    Attempt to extract text from PDF using PyPDFLoader.
    If extracted text is too short, assume PDF is scanned and apply OCR.
    """
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    extracted_text = " ".join([doc.page_content for doc in docs]).strip()

    if len(extracted_text) < 50:
        print(f"📄 The PDF '{filename}' appears to be scanned, using OCR via pdf2image.")
        try:
            pages = convert_from_path(file_path, dpi=300)
            ocr_text = ""
            for page in pages:
                ocr_text += pytesseract.image_to_string(page) + "\n"
            if ocr_text.strip():
                return [Document(page_content=ocr_text, metadata={"source": filename, "type": "pdf-ocr"})]
            else:
                print(f"❌ No text extracted by OCR for '{filename}'.")
                return []
        except Exception as e:
            print(f"❌ Error during OCR for '{filename}': {e}")
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
                print(f"Error loading PDF '{filename}': {e}")

        elif lower_name.endswith((".png", ".jpg", ".jpeg")):
            try:
                image = Image.open(file_path).convert("L")
                text = pytesseract.image_to_string(image)
                documents.append(Document(page_content=text, metadata={"source": filename, "type": "image"}))
            except Exception as e:
                print(f"Error processing image '{filename}': {e}")

        elif lower_name.endswith((".py", ".js", ".java", ".c", ".cpp", ".txt")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                documents.append(Document(page_content=text, metadata={"source": filename, "type": "text"}))
            except Exception as e:
                print(f"Error loading text file '{filename}': {e}")

        else:
            print(f"Unsupported format for file '{filename}'")
    return documents

if __name__ == "__main__":
    docs = load_documents("data")
    print("Documents loaded:", len(docs))

