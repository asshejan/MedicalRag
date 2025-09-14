from fastapi import APIRouter
from app.schemas.tutor import TutorQuestionRequest, TutorAnswerResponse
from app.services.tutor_service import MedicalAITutorService

router = APIRouter(prefix="/tutor", tags=["Medical AI Tutor"])
tutor_service = MedicalAITutorService()

@router.post("/ask", response_model=TutorAnswerResponse)
def ask_tutor(request: TutorQuestionRequest):
    answer = tutor_service.answer_question(request.question)
    return TutorAnswerResponse(answer=answer)
