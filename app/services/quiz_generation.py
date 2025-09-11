def generate_flash_cards(prompt: str, num_cards: int = 10) -> list:
    """
    Generate flash cards using OpenAI. Each card is a dict with 'Question' and 'Answer' (short answer).
    """
    client = openai.OpenAI()
    flash_prompt = (
        prompt +
        f" Generate {num_cards} flash cards. Each should be a JSON object with keys 'Question' and 'Answer' (answer must be short). Return a JSON array."
        "\nExample: [\n  {\"Question\": \"What is the capital of France?\", \"Answer\": \"Paris\"}\n]"
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": flash_prompt}],
        max_tokens=1024,
        temperature=0.7,
    )
    text = response.choices[0].message.content
    import json
    try:
        cards = json.loads(text)
    except Exception:
        import re
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            cards = json.loads(match.group(0))
        else:
            cards = []
    return cards

import openai
import os
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import json

def generate_quiz_questions(context: str, num_questions: int = 5, difficulty: str = "basic", qtype: str = "mcq") -> List[dict]:
    prompt = f"""
    You are a medical quiz generator. Based on the following context, generate {num_questions} {difficulty} multiple choice questions (MCQ) with 4 options each and the correct answer.\nContext:\n{context}\n
    Return the result as a JSON array, where each question is an object with the following keys:\n
    - question: the question text\n    - options: a list of 4 options (a, b, c, d)\n    - answer: the correct option letter (e.g., 'c')\n    - explanation: a short explanation for the answer\n
    Example:\n[
      {{
        "question": "Which of the following is NOT a social determinant of health?",
        "options": ["a.Education", "b.Employment", "c.Genetics", "d.Housing"],
        "answer": "c",
        "explanation": "Genetics is not considered a social determinant of health."
      }}
    ]
    """
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.7,
    )
    # Parse response as JSON
    text = response.choices[0].message.content
    try:
        questions = json.loads(text)
    except Exception:
        # fallback: try to extract JSON from text
        import re
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            questions = json.loads(match.group(0))
        else:
            raise ValueError("Could not parse quiz questions as JSON.")
    return questions
