from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.genai.errors import ServerError
import json

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static folder
app.mount("/pages", StaticFiles(directory="static"), name="static")

client = genai.Client()

# Data models
class PatientInfo(BaseModel):
    birth_year: int
    birth_place: str
    hobbies: list[str]

# UPDATED: The new Multiple Choice Quiz structure
class TriviaMCQ(BaseModel):
    question: str
    options: list[str]
    correct_answer: str

class GameAssets(BaseModel):
    music_cues: list[str]
    local_landmarks: list[str]
    trivia_mcq: list[TriviaMCQ] # Updated to use the MCQ structure

@app.post("/generate-game")
def generate_game(patient: PatientInfo):
    formative_era = patient.birth_year + 15
    prompt = f"Patient details: Born in {patient.birth_year}, grew up in {patient.birth_place}. Hobbies: {', '.join(patient.hobbies)}. Target Era: {formative_era}."

    try:
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="You are a reminiscence therapy expert. Generate historical and cultural game assets. For the trivia, generate multiple-choice questions with exactly 4 options (1 correct, 3 plausible but incorrect historical distractors from their era).",
                response_mime_type="application/json",
                response_schema=GameAssets, 
                temperature=0.7,
            ),
        )
        return json.loads(response.text)
        
    except ServerError:
        raise HTTPException(
            status_code=503, 
            detail="The AI server is currently experiencing high demand. Please wait a few seconds and try again."
        )
    except ClientError as e:
        # <-- Catch the 429 Quota Error here
        if e.code == 429:
            raise HTTPException(
                status_code=429, 
                detail="Daily AI quota exceeded. Please try again tomorrow or update your API key."
            )
        # Catch any other client errors
        raise HTTPException(
            status_code=400, 
            detail=f"API Error: {e.message}"
        )