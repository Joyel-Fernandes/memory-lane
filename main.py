import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static folder so FastAPI serves your HTML pages
app.mount("/pages", StaticFiles(directory="static"), name="static")

# 1. Parse the comma-separated keys from the environment variables
raw_keys = os.getenv("GEMINI_API_KEYS", "")
api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]

# 2. Fallback for local testing if you are using a standard single key variable
if not api_keys and os.getenv("GEMINI_API_KEY"):
    api_keys = [os.getenv("GEMINI_API_KEY")]

# Data models
class PatientInfo(BaseModel):
    birth_year: int
    birth_place: str
    hobbies: list[str]

class TriviaMCQ(BaseModel):
    question: str
    options: list[str]
    correct_answer: str

class GameAssets(BaseModel):
    music_cues: list[str]
    local_landmarks: list[str]
    trivia_mcq: list[TriviaMCQ]

@app.post("/generate-game")
def generate_game(patient: PatientInfo):
    formative_era = patient.birth_year + 15
    prompt = f"Patient details: Born in {patient.birth_year}, grew up in {patient.birth_place}. Hobbies: {', '.join(patient.hobbies)}. Target Era: {formative_era}."

    if not api_keys:
        raise HTTPException(status_code=500, detail="No API keys configured.")

    # 3. Loop through available keys until one succeeds
    for i, key in enumerate(api_keys):
        try:
            client = genai.Client(api_key=key)
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
            
        except ClientError as e:
            # Catch 429 Quota Exhausted limits specifically
            if getattr(e, 'code', None) == 429 or "RESOURCE_EXHAUSTED" in str(e):
                print(f"Key {i+1} quota exhausted. Trying next key...")
                continue # Skip to the next key in the array
            else:
                # Raise other client errors (like 400 Bad Request) immediately
                raise HTTPException(status_code=getattr(e, 'code', 400), detail=str(e))
                
        except ServerError:
            # Keep your existing 503 handling logic
            raise HTTPException(
                status_code=503, 
                detail="The AI server is currently experiencing high demand. Please wait a few seconds and try again."
            )

    # 4. If the loop completes without a successful return, all quotas are drained
    raise HTTPException(
        status_code=429, 
        detail="All Gemini API keys have exceeded their free-tier quota. Please try again later."
    )