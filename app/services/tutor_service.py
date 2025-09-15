import os
import uuid
from typing import Tuple, List, Optional
from datetime import datetime
from app.services.rag_pipeline_pinecone import RAGPipelinePinecone
from openai import OpenAI
from app.schemas.tutor import ConversationExchange, SessionSummary

class MedicalAITutorService:
    def __init__(self):
        self.rag = RAGPipelinePinecone(index_name="medical")
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = "gpt-3.5-turbo"  # Default model
        # Dictionary to store active sessions with their conversation history
        # Format: {session_id: {"user_id": user_id, "conversations": [ConversationExchange], "start_time": datetime, "last_activity": datetime}}
        self.active_sessions = {}
        # Maximum number of exchanges to keep in memory for context
        self.max_context_exchanges = 3

    def answer_question(self, question: str, user_id: str = None, session_id: str = None) -> tuple[str, str, bool]:
        # Check if this is a new session or continuing session
        is_new_session = False
        if not session_id:
            # Generate new session ID for new conversation
            session_id = f"session_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            is_new_session = True
        elif session_id not in self.active_sessions:
            # Session ID provided but not found in active sessions
            is_new_session = True
        
        # Initialize or update session data
        current_time = datetime.now()
        if is_new_session:
            # Create new session
            self.active_sessions[session_id] = {
                "user_id": user_id,
                "conversations": [],
                "start_time": current_time,
                "last_activity": current_time
            }
            # Check if returning user - get previous session summaries
            user_context = ""
            if user_id:
                # Try to retrieve previous session summaries for this user
                previous_summaries = self.get_session_summaries(user_id, limit=2)
                if previous_summaries:
                    user_context = "Previous session topics: " + ", ".join([s.text for s in previous_summaries])
        else:
            # Update existing session's last activity time
            self.active_sessions[session_id]["last_activity"] = current_time
        
        # Retrieve relevant context from vector DB
        print(f"Processing question: {question}")
        if user_id:
            print(f"User ID: {user_id}, Session ID: {session_id}")
        
        # Retrieve medical knowledge from shared knowledge base (no user filter)
        medical_context_chunks = self.rag.retrieve(question, top_k=5)
        
        # Get conversation history for context
        conversation_context = ""
        if session_id in self.active_sessions and self.active_sessions[session_id]["conversations"]:
            # Get the last few exchanges for context
            recent_exchanges = self.active_sessions[session_id]["conversations"][-self.max_context_exchanges:]
            if recent_exchanges:
                conversation_context = "\nRecent conversation:\n"
                for i, exchange in enumerate(recent_exchanges):
                    conversation_context += f"Student: {exchange.question}\nTutor: {exchange.answer}\n"
        
        # Combine medical knowledge with user conversation context
        context_chunks = medical_context_chunks
        
        # First, check if this is a medical-related question
        medical_check_prompt = (
            f"Determine if the following question is related to medical topics, healthcare, anatomy, physiology, "
            f"pathology, pharmacology, or any medical education content. Answer with 'YES' if medical-related, 'NO' if not.\n\n"
            f"Question: {question}\n\n"
            f"Answer (YES/NO):"
        )
        
        medical_response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": medical_check_prompt}],
            max_tokens=10,
            temperature=0.1,
        )
        
        is_medical = medical_response.choices[0].message.content.strip().upper().startswith('YES')
        
        if not is_medical:
            answer = ("This appears to be outside my medical focus. I specialize "
                     "in medical topics like anatomy, physiology, pathology, pharmacology, and other "
                     "healthcare subjects. What medical concept can I help you understand?")
            
            return answer, session_id
        
        # If it's medical but no context found
        if not context_chunks:
            answer = ("This is a medical topic I can help with, but "
                     "I don't have specific information about this in my current knowledge base. "
                     "What specific aspect would you like to explore? "
                     "If you have relevant medical documents, you can upload them to enhance my knowledge.")
            
            return answer, session_id
        
        context = "\n".join(context_chunks)
        print(f"Retrieved {len(context_chunks)} medical knowledge chunks for answering")
        
        # Build context description
        context_description = "Context from medical literature (shared knowledge base):\n"
        
        # Build personalized context for the user
        user_style_info = ""
        
        if user_id:
            user_context = f"This is a conversation with user {user_id}. "
        
        prompt = (
            f"You are a medical tutor having a natural conversation with a medical student. "
            f"You're knowledgeable and focused on providing accurate information. "
            f"{user_context}{user_style_info}"
            f"Important instructions:\n"
            f"1. Use the medical knowledge from the shared knowledge base to answer medical questions\n"
            f"2. Use the user's conversation history ONLY for conversational context and continuity\n"
            f"3. If the medical knowledge doesn't contain enough information, "
            f"acknowledge this and explain what specific information is missing\n"
            f"4. Never make assumptions or add information beyond what's in the medical knowledge\n"
            f"5. Be conversational but direct - use natural language without unnecessary enthusiasm or greetings\n"
            f"6. If the medical knowledge is relevant but incomplete, provide what information you can\n"
            f"7. Answer directly - avoid phrases like 'Based on the information provided' or 'According to the context'\n"
            f"8. Present the medical information clearly and professionally\n"
            f"9. If you see previous conversations in the context, reference them naturally to provide continuity\n"
            f"10. If this is a follow-up question, maintain context from the previous exchanges\n"
            f"10. Avoid overused phrases like 'That's a great question!' or 'That's interesting!'\n"
            f"11. Keep responses focused on the medical content without unnecessary commentary\n\n"
            f"{context_description}{context}\n\n"
            f"{conversation_context}\n"
            f"Student's Question: {question}\n\n"
            f"Tutor's Response (use medical knowledge + conversational context):"
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,  # Increased for more conversational responses
            temperature=0.7,  # Higher temperature for more natural, varied responses
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Store the conversation exchange
        if session_id in self.active_sessions:
            exchange = ConversationExchange(
                question=question,
                answer=answer,
                timestamp=datetime.now()
            )
            self.active_sessions[session_id]["conversations"].append(exchange)
        
        return answer, session_id, is_new_session
    
    def get_conversation_history(self, session_id: str) -> List[ConversationExchange]:
        """Retrieve conversation history for a specific session"""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]["conversations"]
        return []
    
    def end_session(self, session_id: str) -> Tuple[str, Optional[str]]:
        """End a session and generate a summary"""
        if session_id not in self.active_sessions:
            return "error", "Session not found"
        
        # Get the conversation history
        conversations = self.active_sessions[session_id]["conversations"]
        if not conversations:
            return "ended", "No conversation to summarize"
        
        # Generate a summary of the conversation
        summary = self._generate_session_summary(session_id, conversations)
        
        # Store the summary in Pinecone for future reference
        self._store_session_summary(session_id, summary)
        
        # Remove the session from active sessions
        user_id = self.active_sessions[session_id]["user_id"]
        del self.active_sessions[session_id]
        
        return "ended", summary
    
    def _generate_session_summary(self, session_id: str, conversations: List[ConversationExchange]) -> str:
        """Generate a summary of the conversation using OpenAI"""
        # Prepare the conversation text for summarization
        conversation_text = ""
        for exchange in conversations:
            conversation_text += f"Student: {exchange.question}\nTutor: {exchange.answer}\n\n"
        
        # Create a prompt for summarization
        prompt = (
            f"Below is a conversation between a medical student and an AI tutor. \n\n"
            f"{conversation_text}\n\n"
            f"Please provide a concise summary of this conversation, highlighting:\n"
            f"1. The main topics discussed\n"
            f"2. Key questions asked by the student\n"
            f"3. Important information provided by the tutor\n\n"
            f"Summary:"
        )
        
        # Generate summary using OpenAI
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.5,
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Session ended. Summary generation failed."
    
    def _store_session_summary(self, session_id: str, summary_text: str) -> None:
        """Store the session summary in Pinecone for future reference"""
        if session_id not in self.active_sessions:
            return
        
        user_id = self.active_sessions[session_id]["user_id"]
        if not user_id:
            return
        
        # Extract topics from the summary
        topics = self._extract_topics_from_summary(summary_text)
        
        # Create a SessionSummary object
        summary = SessionSummary(
            session_id=session_id,
            user_id=user_id,
            text=summary_text,
            timestamp=datetime.now(),
            topics=topics
        )
        
        # Store the summary in Pinecone with special metadata
        metadata = {
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": summary.timestamp.isoformat(),
            "is_summary": True,
            "summary_type": "session_summary",
            "topics": ",".join(topics)
        }
        
        # Add the summary to the vector database
        self.rag.add_document(
            text=summary_text,
            metadata=metadata,
            document_id=f"summary_{session_id}"
        )
    
    def _extract_topics_from_summary(self, summary_text: str) -> List[str]:
        """Extract main topics from the summary using OpenAI"""
        prompt = (
            f"Based on the following conversation summary, extract 3-5 main medical topics discussed. \n"
            f"Return ONLY a comma-separated list of topics, with no additional text.\n\n"
            f"Summary: {summary_text}\n\n"
            f"Topics:"
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.3,
            )
            
            topics_text = response.choices[0].message.content.strip()
            return [topic.strip() for topic in topics_text.split(",") if topic.strip()]
        except Exception as e:
            print(f"Error extracting topics: {e}")
            return ["medical"]
    
    def get_session_summaries(self, user_id: str, session_id: Optional[str] = None, limit: int = 10) -> List[SessionSummary]:
        """Retrieve session summaries for a user from Pinecone"""
        # Prepare the query to find summaries
        query = f"session summaries for user {user_id}"
        filter_dict = {
            "user_id": user_id,
            "is_summary": True,
            "summary_type": "session_summary"
        }
        
        if session_id:
            filter_dict["session_id"] = session_id
        
        # Query Pinecone for summaries
        results = self.rag.retrieve_with_filter(query, filter_dict, top_k=limit)
        
        # Convert results to SessionSummary objects
        summaries = []
        for text, metadata in results:
            if not metadata:
                continue
                
            try:
                topics = metadata.get("topics", "").split(",") if metadata.get("topics") else []
                timestamp_str = metadata.get("timestamp")
                timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()
                
                summary = SessionSummary(
                    session_id=metadata.get("session_id", ""),
                    user_id=metadata.get("user_id", ""),
                    text=text,
                    timestamp=timestamp,
                    topics=topics
                )
                summaries.append(summary)
            except Exception as e:
                print(f"Error parsing summary: {e}")
        
        return summaries
