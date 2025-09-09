# MedStudentAssistant

A FastAPI-based backend for uploading study files and generating quizzes using Retrieval-Augmented Generation (RAG).

## Features
- Upload PDF, image, video, or CSV files
- Extracts content and adds to a RAG pipeline
- Generate quizzes (MCQ, etc.) from uploaded content
- Adjustable quiz parameters: number of questions, difficulty, type

## API Usage

### 1. Generate Quiz (Upload + Quiz Generation)
**Endpoint:** `POST /generate-quiz/`

**Request:** `multipart/form-data`
- `files`: One or more files to upload (PDF, image, video, CSV)
- `query`: (string) Topic or question for quiz generation
- `num_questions`: (integer, default: 5) Number of questions
- `difficulty`: (string, default: "basic") Difficulty level
- `qtype`: (string, default: "mcq") Quiz type

**Example (Swagger UI):**
- Upload file(s) and fill in quiz parameters in the `/generate-quiz/` form

### 2. Get Quiz
**Endpoint:** `GET /quiz/{quiz_id}`
- Returns the generated quiz by ID

## Setup

1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the server**
   ```bash
   uvicorn app.main:app --reload
   ```
4. **Open docs**
   - Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Project Structure
```
app/
  main.py           # FastAPI app and endpoints
  services/         # File processing, RAG, quiz generation logic
  uploads/          # Uploaded files
  models/, routers/, schemas/  # (Extend as needed)
requirements.txt    # Python dependencies
```

## License
MIT
