import os
from app.services.rag_pipeline_pinecone import RAGPipelinePinecone
from openai import OpenAI

class MedicalAITutorService:
    def __init__(self):
        self.rag = RAGPipelinePinecone(index_name="medical")
        self.llm = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def answer_question(self, question: str) -> str:
        # Retrieve relevant context from vector DB
        print(f"Processing question: {question}")
        context_chunks = self.rag.retrieve(question, top_k=5)  # Increased top_k for better context
        
        # First, check if this is a medical-related question
        medical_check_prompt = (
            f"Determine if the following question is related to medical topics, healthcare, anatomy, physiology, "
            f"pathology, pharmacology, or any medical education content. Answer with 'YES' if medical-related, 'NO' if not.\n\n"
            f"Question: {question}\n\n"
            f"Answer (YES/NO):"
        )
        
        medical_response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": medical_check_prompt}],
            max_tokens=10,
            temperature=0.1,
        )
        
        is_medical = medical_response.choices[0].message.content.strip().upper().startswith('YES')
        
        if not is_medical:
            return ("I'm your medical tutor, and I'm here to help you with medical-related questions and concepts! "
                   "I specialize in topics like anatomy, physiology, pathology, pharmacology, and other medical subjects. "
                   "If you have any questions about medical topics, I'd be happy to help you understand them better. "
                   "What medical concept would you like to explore?")
        
        # If it's medical but no context found
        if not context_chunks:
            return ("I'm your medical tutor, and while I don't have specific information about this exact topic "
                   "in my current knowledge base, I'd be happy to help you understand the general concepts. "
                   "Could you provide more details about what specific aspect you'd like to know about? "
                   "Also, if you have relevant medical documents, you can upload them to expand my knowledge base.")
        
        context = "\n".join(context_chunks)
        print(f"Retrieved {len(context_chunks)} context chunks for answering")
        
        prompt = (
            f"You are an empathetic medical tutor with extensive experience in teaching medical students. "
            f"Your role is to help students understand medical concepts clearly and accurately. "
            f"Important instructions:\n"
            f"1. ONLY use the information provided in the context below to answer the question\n"
            f"2. If the context doesn't contain enough information to fully answer the question, "
            f"kindly acknowledge this and explain what specific information is missing\n"
            f"3. Never make assumptions or add information beyond what's in the context\n"
            f"4. Use a supportive and encouraging tone\n"
            f"5. If the context is relevant but incomplete, provide what information you can and suggest what else might be needed\n"
            f"6. Answer directly and naturally - avoid phrases like 'Based on the information provided' or 'According to the context'\n"
            f"7. Present the information as if you're explaining it directly from your knowledge\n\n"
            f"Context from medical literature:\n{context}\n\n"
            f"Student's Question: {question}\n\n"
            f"Tutor's Response:"
        )
        
        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,  # Increased for more detailed responses
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
