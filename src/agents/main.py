#!/usr/bin/env python
import sys
import os
import shutil
import asyncio
import threading
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import existing logic
from agents.crew import SpeechCrew, FileCrew, Nearbyhospitals, general_que
from agents.speech_to_text import transcribe_audio
from agents.buying_agent import run_comparison
from datetime import datetime
from agents.Calling import start_call
from agents.face_recognize import face_recognice
from PIL import Image
import pytesseract

# Initialize FastAPI
app = FastAPI(title="Medical Agent API", version="1.0")

# Allow CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Request Bodies ---

class MedicineComparisonRequest(BaseModel):
    medicines: List[str]

class CallRequest(BaseModel):
    number: str
    disease: str
    doctor_name: str
    availability: str

class LocationRequest(BaseModel):
    city: str
    state: str
    country: str = "India"
    limit: int = 5

# --- Helper Functions ---

def save_upload_file(upload_file: UploadFile, destination: str):
    try:
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()

def extract_text_from_image(image_path: str) -> str:
    """Re-implementation of file_to_text logic to avoid input() blocking"""
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang='eng')
    return text

# --- Endpoints ---

@app.get("/")
def health_check():
    return {"status": "active", "message": "Medical Agent API is running"}

@app.post("/consultation/speech")
async def speech_consultation(audio_file: UploadFile = File(...)):
    """
    Choice 1: Upload an audio file (wav/mp3) for consultation.
    Returns the consultation result.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    temp_filename = f"temp_{audio_file.filename}"
    
    try:
        # 1. Save and Transcribe
        save_upload_file(audio_file, temp_filename)
        statement = transcribe_audio(temp_filename)
        
        # 2. Run Crew
        inputs = {"statement": f"[{timestamp}] {statement}"}
        result = SpeechCrew().crew().kickoff(inputs=inputs)
        
        # Return result and the medicine name for potential price comparison
        return {
            "consultation_result": result.raw,
            "medicine_name": result.pydantic.name if hasattr(result, 'pydantic') and hasattr(result.pydantic, 'name') else None,
            "details": result.pydantic.dict() if hasattr(result, 'pydantic') else {}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.post("/consultation/document")
async def document_consultation(file: UploadFile = File(...)):
    """
    Choice 2: Upload a medical document (image) for extraction.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    temp_filename = f"temp_{file.filename}"

    try:
        # 1. Save and Extract Text
        save_upload_file(file, temp_filename)
        statement = extract_text_from_image(temp_filename)
        
        # 2. Run Crew
        inputs = {"statement": f"[{timestamp}] {statement}"}
        result = FileCrew().crew().kickoff(inputs=inputs)

        return {
            "extraction_result": result.raw,
            "medicine_name": result.pydantic.name if hasattr(result, 'pydantic') and hasattr(result.pydantic, 'name') else None,
            "details": result.pydantic.dict() if hasattr(result, 'pydantic') else {}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.post("/medicine/compare")
async def compare_prices(request: MedicineComparisonRequest):
    """
    Follow-up: Run price comparison for a list of medicines.
    """
    try:
        # Assuming run_comparison is async or we run it in thread
        # The provided buying_agent.py has 'async def run_comparison'
        best_deal = await run_comparison(request.medicines)
        return {"best_deal": best_deal}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/call/schedule")
def schedule_call(request: CallRequest, background_tasks: BackgroundTasks):
    """
    Choice 3: Schedule/Start a call.
    Runs in a background thread because start_call blocks with uvicorn.
    """
    try:
        # We run this in a separate thread so it doesn't block the API response
        # Note: start_call internally spins up a Uvicorn server on port 8000
        # Ensure this API runs on a different port (e.g., 8080)
        thread = threading.Thread(
            target=start_call, 
            args=(request.number, request.disease, request.doctor_name, request.availability),
            daemon=True
        )
        thread.start()
        
        return {"status": "initiated", "message": f"Calling process started for {request.number}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/hospitals/nearby")
def find_hospitals(location: Optional[LocationRequest] = None):
    """
    Choice 4: Find nearby hospitals.
    If location is provided, uses that.
    If not, triggers server-side Face Recognition to get profile (Original Logic).
    """
    try:
        if location:
            city = location.city
            state = location.state
        else:
            # Trigger Facial Recognition on Server Camera
            # Note: This requires the server to have a webcam/display access
            user_profile = face_recognice()
            if not user_profile:
                raise HTTPException(status_code=404, detail="User not recognized and no location provided.")
            city = user_profile.get("City")
            state = user_profile.get("State")

        inputs = {
            "city": city,
            "state": state,
            "country": "India",
            "limit": 5
        }

        result = Nearbyhospitals().crew().kickoff(inputs=inputs)
        return {"hospitals": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/general-query")
async def general_health_query(audio_file: UploadFile = File(...)):
    """
    Choice 5: General health query via voice.
    """
    temp_filename = f"temp_query_{audio_file.filename}"
    try:
        save_upload_file(audio_file, temp_filename)
        question = transcribe_audio(temp_filename)
        
        result = general_que().crew().kickoff(inputs={"question": question})
        return {"answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)