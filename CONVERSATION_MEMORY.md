# Conversation Memory Features

## Overview

This document describes the conversation memory features implemented in the Medical Student Assistant application. These features enhance the user experience by maintaining context between messages, providing personalized greetings, and storing session summaries for future reference.

## Features Implemented

### 1. Short-term Memory for Conversations

- The system now maintains context between messages within a session
- Each conversation is stored with user_id and session_id in Pinecone
- Follow-up questions can reference previous exchanges in the same session
- The last 3 exchanges are used to provide context for new questions

### 2. Session Greetings

- New sessions start with a friendly greeting
- Returning users receive personalized greetings that reference previous session topics
- The system detects whether a session is new or continuing

### 3. Session Summaries

- When a session ends, a summary is automatically generated using GPT-3.5-turbo
- Summaries are stored in Pinecone with special metadata (is_summary=True, summary_type="session_summary")
- Summaries can be retrieved for any user to provide context for future sessions
- A new endpoint `/session-summaries` allows retrieving session summaries for a user

## API Endpoints

### New Endpoints

- **POST /tutor/end-session**: Ends a session and generates a summary
  - Parameters: `session_id` (query parameter)
  - Returns: Session status and summary

- **GET /tutor/session-summaries**: Retrieves session summaries for a user
  - Parameters: `user_id` (required), `session_id` (optional), `limit` (optional, default=10)
  - Returns: List of session summaries

### Updated Endpoints

- **POST /tutor/ask**: Now accepts an optional `session_id` parameter to maintain context
  - If not provided, a new session ID is generated
  - Returns the session_id and whether it's a new session

## Testing

A test script `test_conversation_memory.py` is provided to verify the functionality:

1. Starts a new conversation (should include greeting)
2. Continues with follow-up questions (should maintain context)
3. Retrieves conversation history
4. Ends the session and generates a summary
5. Starts a new session (should include personalized greeting)
6. Retrieves session summaries

To run the test:

```bash
python test_conversation_memory.py
```

## Implementation Details

- Session tracking is implemented in the `MedicalAITutorService` class
- Active sessions are tracked in memory with user_id, start_time, last_activity, and exchanges count
- The RAG pipeline was extended to support storing and retrieving summaries
- Metadata filtering is used to distinguish between regular conversations and summaries