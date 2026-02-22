import fitz                          # PyMuPDF
import io
from docx import Document            # python-docx

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    texts = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        texts.append((text, page_num + 1))
    return texts

def extract_text_from_txt(file):
    text = file.read().decode("utf-8")
    return [(text, 1)]

def extract_text_from_docx(file):
    doc = Document(io.BytesIO(file.read()))
    full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    return [(full_text, 1)]

def load_file(file):
    if file.name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif file.name.endswith(".txt"):
        return extract_text_from_txt(file)
    elif file.name.endswith(".docx"):
        return extract_text_from_docx(file)
    else:
        raise ValueError(f"Unsupported file type: {file.name}")
