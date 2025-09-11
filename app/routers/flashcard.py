

from fastapi import APIRouter, UploadFile, File
from app.schemas.flashcard import FlashCardRequest, FlashCardResponse
from app.services.flashcard_service import handle_flashcard_generation

router = APIRouter()


# Flash card generation endpoint

@router.post("/flash-card/", response_model=dict)
async def generate_flash_card(
    file: UploadFile = File(None),
    subject: str = None,
    chapter: str = None,
    topic: str = None,
    num_cards: int = 10
):
    """
    Generate flash cards as a list of dicts with keys 'Question' and 'Answer' (short answer max 5 words).
    """
    req = FlashCardRequest(subject=subject, chapter=chapter, topic=topic, num_cards=num_cards)
    flash_cards = handle_flashcard_generation(file=file, request=req)
    return {"flash_cards": flash_cards}