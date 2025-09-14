from pydantic import BaseModel

class TutorQuestionRequest(BaseModel):
    question: str

class TutorAnswerResponse(BaseModel):
    answer: str
