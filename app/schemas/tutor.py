from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime

class TutorQuestionRequest(BaseModel):
    question: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class TutorAnswerResponse(BaseModel):
    answer: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    is_new_session: Optional[bool] = False

class ConversationExchange(BaseModel):
    question: str
    answer: str
    timestamp: datetime = datetime.now()

class ConversationHistoryResponse(BaseModel):
    conversations: List[ConversationExchange]
    user_id: str
    session_id: str
    total_count: int

class EndSessionRequest(BaseModel):
    session_id: str

class EndSessionResponse(BaseModel):
    status: str
    summary: Optional[str] = None

class SessionSummary(BaseModel):
    session_id: str
    user_id: str
    text: str
    timestamp: datetime
    topics: List[str] = []

class SessionSummariesResponse(BaseModel):
    summaries: List[SessionSummary]
    user_id: str
    total_count: int
