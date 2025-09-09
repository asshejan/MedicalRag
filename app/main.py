

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import os
from typing import List
from app.services.file_processing import extract_text
from app.services.rag_pipeline import RAGPipeline
from app.services.quiz_generation import generate_quiz_questions
import uuid

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory store for quizzes (for demo)
QUIZ_STORE = {}
rag_pipeline = RAGPipeline()

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI base setup!"}




# Combined upload and generate quiz endpoint
@app.post("/generate-quiz/")
async def generate_quiz(
    files: list[UploadFile] = File(...),
    query: str = Form(...),
    num_questions: int = Form(5),
    difficulty: str = Form("basic"),
    qtype: str = Form("mcq")
):
    allowed_types = [
        "application/pdf", "text/csv",
        "image/jpeg", "image/png", "image/jpg",
        "video/mp4", "video/quicktime"
    ]
    all_text = []
    for file in files:
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail=f"File type {file.content_type} not allowed.")
        file_id = str(uuid.uuid4())
        file_location = os.path.join(UPLOAD_DIR, file_id + "_" + file.filename)
        with open(file_location, "wb") as f:
            f.write(await file.read())
        # Extract text and add to RAG
        text = extract_text(file_location, file.content_type)
        rag_pipeline.add_document(text)
        all_text.append(text)
    # Retrieve relevant context from RAG (limit to top 5 chunks)
    context_list = rag_pipeline.retrieve(query, top_k=5)
    # Further limit context to avoid token overflow
    max_context_length = 3000  # characters
    context = "\n".join(context_list)
    if len(context) > max_context_length:
        context = context[:max_context_length]
    questions = generate_quiz_questions(context, num_questions, difficulty, qtype)
    quiz_id = str(uuid.uuid4())
    QUIZ_STORE[quiz_id] = questions
    return {"quiz_id": quiz_id, "questions": questions}


# Get quiz endpoint
@app.get("/quiz/{quiz_id}")
def get_quiz(quiz_id: str):
    questions = QUIZ_STORE.get(quiz_id)
    if not questions:
        raise HTTPException(status_code=404, detail="Quiz not found.")
    return {"quiz_id": quiz_id, "questions": questions}
