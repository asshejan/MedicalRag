from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.rag_pipeline_pinecone import RAGPipelinePinecone
from app.services.file_processing import extract_text
import os
import uuid

router = APIRouter(prefix="/tutor", tags=["Medical AI Tutor"])
rag_pipeline = RAGPipelinePinecone(index_name="medical")  # Match the index name in Pinecone

@router.post("/upload-knowledge")
async def upload_knowledge_file(file: UploadFile = File(...)):
    allowed_types = [
        "application/pdf", "text/csv",
        "image/jpeg", "image/png", "image/jpg",
        "video/mp4", "video/quicktime"
    ]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"File type {file.content_type} not allowed.")
    file_id = str(uuid.uuid4())
    file_location = os.path.join(os.path.dirname(__file__), "../../uploads", file_id + "_" + file.filename)
    os.makedirs(os.path.dirname(file_location), exist_ok=True)
    with open(file_location, "wb") as f:
        f.write(await file.read())
    try:
        print(f"Extracting text from {file.filename}...")
        text = extract_text(file_location, file.content_type)
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the file")
        
        # Check if document already exists in the index
        print(f"Checking if document already exists...")
        if rag_pipeline.document_exists(text):
            # Clean up the file since we don't need it
            try:
                os.remove(file_location)
            except Exception as e:
                print(f"Warning: Could not delete file {file_location}: {e}")
                
            return {
                "message": "This document or very similar content already exists in the knowledge base.",
                "status": "duplicate_document",
                "filename": file.filename
            }
        
        print(f"Adding document to RAG pipeline...")
        
        # Check index health before adding document
        health_check = rag_pipeline.check_index_health()
        print(f"Index health check: {health_check}")
        
        doc_stats = rag_pipeline.add_document(text)
        print(f"Successfully processed document: {doc_stats}")
        
        # Verify the document was added by checking index stats
        index_stats = rag_pipeline.get_index_stats()
        print(f"Index stats after upload: {index_stats}")
        
        try:
            os.remove(file_location)
        except Exception as e:
            print(f"Warning: Could not delete file {file_location}: {e}")
            
        return {
            "message": "File uploaded and added to tutor knowledge base.",
            "stats": doc_stats,
            "index_stats": index_stats,
            "health_check": health_check
        }
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/index-status")
def get_index_status():
    """Get the current status of the medical knowledge index."""
    try:
        health_check = rag_pipeline.check_index_health()
        index_stats = rag_pipeline.get_index_stats()
        
        return {
            "index_name": "medical",
            "health_check": health_check,
            "index_stats": index_stats,
            "message": "Index status retrieved successfully"
        }
    except Exception as e:
        print(f"Error getting index status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
