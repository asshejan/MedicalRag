

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import os
from typing import List
from app.services.file_processing import extract_text
from app.services.rag_pipeline import RAGPipeline
from app.services.quiz_generation import generate_quiz_questions, generate_flash_cards
import uuid

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)





from app.routers import quiz, flashcard
app = FastAPI()
app.include_router(quiz.router)
app.include_router(flashcard.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI base setup!"}



