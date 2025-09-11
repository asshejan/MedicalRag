

from fastapi import FastAPI
import os

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)





from app.routers import quiz, flashcard

app = FastAPI(
    title="Medical Student Assistant",
    description="AI-powered medical study assistant with RAG capabilities",
    version="1.0.0"
)

app.include_router(quiz.router)
app.include_router(flashcard.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to Medical Student Assistant API!"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "API is running"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)



