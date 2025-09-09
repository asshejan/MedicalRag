import os
import fitz  # PyMuPDF
import pandas as pd
import pytesseract
from PIL import Image
from moviepy.editor import VideoFileClip
import tempfile

# Extract text from PDF

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Extract text from CSV

def extract_text_from_csv(file_path: str) -> str:
    df = pd.read_csv(file_path)
    return df.to_string()

# Extract text from image using OCR

def extract_text_from_image(file_path: str) -> str:
    img = Image.open(file_path)
    return pytesseract.image_to_string(img)

# Extract text from video using speech-to-text (placeholder)
def extract_text_from_video(file_path: str) -> str:
    # Placeholder: Implement speech-to-text with a real model/service
    return "Speech-to-text not implemented."

# Main dispatcher

def extract_text(file_path: str, content_type: str) -> str:
    if content_type == "application/pdf":
        return extract_text_from_pdf(file_path)
    elif content_type == "text/csv":
        return extract_text_from_csv(file_path)
    elif content_type in ["image/jpeg", "image/png", "image/jpg"]:
        return extract_text_from_image(file_path)
    elif content_type in ["video/mp4", "video/quicktime"]:
        return extract_text_from_video(file_path)
    else:
        return "Unsupported file type."
