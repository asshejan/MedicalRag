import os
import uuid
from fastapi import UploadFile
from app.services.file_processing import extract_text
from app.services.quiz_generation import generate_flash_cards
from app.schemas.flashcard import FlashCardRequest

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def handle_flashcard_generation(
    file: UploadFile = None,
    request: FlashCardRequest = None
):
    example = """
Example:\n[\n  {\n    \"Question\": \"What deep muscle of the forearm flexor compartment has both its origin and insertion located most distally on the forearm?\",\n    \"Answer\": \"Pronator quadratus\"\n  }\n]\n"""
    if file is not None:
        file_id = str(uuid.uuid4())
        file_location = os.path.join(UPLOAD_DIR, file_id + "_" + file.filename)
        with open(file_location, "wb") as f:
            f.write(file.file.read())
        text = extract_text(file_location, file.content_type)
        try:
            os.remove(file_location)
        except Exception as e:
            print(f"Warning: Could not delete file {file_location}: {e}")
        prompt = f"Generate flash cards from the following content: {text}. Each flash card should have a question and a short (maximum 5 word) answer. Return as a list of JSON objects with keys 'Question' and 'Answer'.\n" + example
    else:
        prompt = f"Generate flash cards for subject: {request.subject}, chapter: {request.chapter}"
        if request.topic:
            prompt += f", topic: {request.topic}"
        prompt += ". Each flash card should have a question and a short(maximum 5 word) answer. Return as a list of JSON objects with keys 'Question' and 'Answer'.\n" + example
    flash_cards = generate_flash_cards(prompt, num_cards=request.num_cards)
    return flash_cards

