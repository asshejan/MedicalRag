from fastapi import APIRouter, HTTPException
from collections import defaultdict, deque
from typing import Tuple, List
from app.schemas.tutor import (
    TutorQuestionRequest,
    TutorAnswerResponse,
)
from app.services.rag_pipeline_pinecone import RAGPipelinePinecone

router = APIRouter(prefix="/tutor", tags=["Medical AI Tutor"])

# Initialize RAG pipeline
rag_pipeline = RAGPipelinePinecone(index_name="medical")

class MedicalAITutorService:
    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline
        # session_id -> deque of (role, message)
        self.memory = defaultdict(lambda: deque(maxlen=10))  # keep last 10 messages

    def answer_question(self, question: str, user_id: str, session_id: str = None) -> Tuple[str, str, bool]:
        """Answer a user question with conversational memory."""
        import uuid
        is_new_session = False

        if not session_id:
            session_id = str(uuid.uuid4())
            is_new_session = True

        # Store user message in memory
        self.memory[session_id].append(("user", question))

        # Retrieve context from RAG
        retrieved_chunks = self.rag.retrieve(query=question, top_k=5, user_id=user_id, session_id=session_id)

        # Build conversational context
        conversation_context = "\n".join([f"{role}: {msg}" for role, msg in self.memory[session_id]])
        context_prompt = f"""
You are a medical education tutor for a medical student. Follow these rules strictly:
- Assume the user is a medical student, not a patient.
- Be educational, concise, and clinically accurate. Explain reasoning and key differentials when relevant.
- Ground answers ONLY in the retrieved knowledge and standard medical knowledge. If the answer is not supported by the retrieved context, say you don't know rather than guessing.
- Do not provide personal medical advice. Frame content academically (e.g., epidemiology, pathophysiology, diagnostics, management frameworks).
- Prefer structured outputs: bullets, short sections, stepwise reasoning.
- If the user asks patient-like questions, respond with teaching content for students (e.g., red flags, diagnostic approach) instead of personalized guidance.

Conversation so far:
{conversation_context}

Retrieved medical context:
{retrieved_chunks}

Now answer the latest question for a medical student audience.
"""
        # Call OpenAI for generation
        response = self.rag._client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": context_prompt}]
        )

        answer = response.choices[0].message.content

        # Store assistant response in memory
        self.memory[session_id].append(("assistant", answer))

        return answer, session_id, is_new_session

    def get_conversation_history(self, session_id: str) -> List[dict]:
        """Return stored conversation history for a session."""
        return [{"role": role, "message": msg} for role, msg in self.memory.get(session_id, [])]

    def end_session(self, session_id: str):
        """Summarize and clear the session memory."""
        conversation = self.get_conversation_history(session_id)
        summary_prompt = f"Summarize this medical tutoring session briefly:\n{conversation}"

        response = self.rag._client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": summary_prompt}]
        )
        summary = response.choices[0].message.content

        # Optionally persist summary in DB
        self.memory.pop(session_id, None)

        return "ended", summary

def end_session(self, session_id: str, user_id: str = None):
    """Summarize and store the session in Pinecone, then clear short-term memory."""
    conversation = self.get_conversation_history(session_id)

    if not conversation:
        return "no_conversation", "No conversation history to summarize."

    # Create summary
    summary_prompt = f"Summarize this medical tutoring session in 5-6 sentences:\n{conversation}"
    response = self.rag._client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": summary_prompt}]
    )
    summary = response.choices[0].message.content

    # Store summary in Pinecone
    try:
        # Embed the summary
        summary_embedding = self.rag._embed_texts([summary])[0]

        # Generate unique summary id
        import uuid
        summary_id = f"summary_{session_id}_{uuid.uuid4().hex[:8]}"

        # Metadata for retrieval later
        metadata = {
            "type": "session_summary",
            "session_id": session_id,
            "user_id": user_id or "unknown",
            "text": summary,
        }

        # Upsert to Pinecone
        self.rag._index.upsert(vectors=[(summary_id, summary_embedding, metadata)])
        print(f"Stored session summary in Pinecone with id {summary_id}")

    except Exception as e:
        print(f"Error storing summary in Pinecone: {e}")

    # Clear short-term memory
    self.memory.pop(session_id, None)

    return "ended", summary

def get_session_summaries(self, user_id: str, session_id: str = None, limit: int = 10):
    """Retrieve stored session summaries for a user from Pinecone."""
    try:
        filter_dict = {"type": "session_summary", "user_id": user_id}
        if session_id:
            filter_dict["session_id"] = session_id

        results = self.rag._index.query(
            vector=[0.0] * 1536,  # dummy vector for filtering only
            top_k=limit,
            include_metadata=True,
            filter=filter_dict
        )

        summaries = []
        for match in results.matches:
            summaries.append({
                "session_id": match.metadata.get("session_id"),
                "text": match.metadata.get("text"),
                "user_id": match.metadata.get("user_id"),
            })

        return summaries
    except Exception as e:
        print(f"Error retrieving session summaries: {e}")
        return []

# Initialize the tutor service
tutor_service = MedicalAITutorService(rag_pipeline)

@router.post("/ask", response_model=TutorAnswerResponse)
async def ask_question(request: TutorQuestionRequest):
    """Ask a question to the medical AI tutor."""
    try:
        answer, session_id, is_new_session = tutor_service.answer_question(
            question=request.question,
            user_id=request.user_id or "anonymous",
            session_id=request.session_id
        )
        
        return TutorAnswerResponse(
            answer=answer,
            user_id=request.user_id,
            session_id=session_id,
            is_new_session=is_new_session
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/conversation-history/{session_id}")
async def get_conversation_history(session_id: str):
    """Return full chronological conversation for a session as role/message pairs."""
    try:
        history = tutor_service.get_conversation_history(session_id)
        # history is already a list of {role, message}
        return {
            "session_id": session_id,
            "history": history,
            "total_count": len(history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
