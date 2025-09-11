from pydantic import BaseModel
from typing import Optional

class FlashCardRequest(BaseModel):
    subject: Optional[str] = None
    chapter: Optional[str] = None
    topic: Optional[str] = None
    num_cards: int = 10

class FlashCardResponse(BaseModel):
    Question: str
    Answer: str
