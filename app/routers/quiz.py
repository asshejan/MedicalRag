
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
import os
import uuid
from app.services.file_processing import extract_text
from app.services.rag_pipeline import RAGPipeline
from app.services.quiz_generation import generate_quiz_questions

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

QUIZ_STORE = {}
rag_pipeline = RAGPipeline()

router = APIRouter()

@router.post("/generate-quiz/")
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
        text = extract_text(file_location, file.content_type)
        rag_pipeline.add_document(text)
        all_text.append(text)
        try:
            os.remove(file_location)
        except Exception as e:
            print(f"Warning: Could not delete file {file_location}: {e}")
    context_list = rag_pipeline.retrieve(query, top_k=5)
    max_context_length = 3000
    context = "\n".join(context_list)
    if len(context) > max_context_length:
        context = context[:max_context_length]
    questions = generate_quiz_questions(context, num_questions, difficulty, qtype)
    quiz_id = str(uuid.uuid4())
    QUIZ_STORE[quiz_id] = questions
    return {"quiz_id": quiz_id, "questions": questions}

@router.get("/quiz/{quiz_id}")
def get_quiz(quiz_id: str):
    questions = QUIZ_STORE.get(quiz_id)
    if not questions:
        raise HTTPException(status_code=404, detail="Quiz not found.")
    return {"quiz_id": quiz_id, "questions": questions}
