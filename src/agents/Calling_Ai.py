from crewai import Agent, Task, Crew, Process
import google.generativeai as genai
import os
from dotenv import load_dotenv
from agents.face_recognize import face_recognice

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

user_details = face_recognice()

def gemini_response(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    print(response.text)
    return response.text

def decide_reply(transcript_text: str,conversation_history: list,slot: str,doctor_name: str,disease: str):
    prompt = f"""
    You are Sarah, a personal assistant calling a dental clinic to book an appointment for your boss.
    
    Current situation: You are on the phone with the clinic receptionist.
    The prev conversation of your with the receptionist is: "{conversation_history}"
    The receptionist just said: "{transcript_text}".
    
    Your Goal: Book an appointment for the slot {slot} for doctor {doctor_name} for disease {disease}.
    
    Guidelines:
    - Analyze the conversation history till now to give the best possible response
    - Keep your responses short as possible (1-2 sentences).
    - Be polite but clear about the preferred dates.
    - Determine the next natural thing to say to the receptionist.

    If the Receptionist ask about deatils of the Boss then use this to give the details {user_details}
    """
    
    try:
        return gemini_response(prompt)
    except Exception as e:
        print("Gemini API error:", e)
        return "I'm sorry, I missed that. Could you repeat the available times?"
    
def summarize_conversation(full_history: list):
    """
    Takes the complete list of conversation turns and generates a summary.
    """
    if not full_history:
        return "No conversation occurred."

    history_text = "\n".join(full_history)
    
    prompt = f"""
    You are an expert note-taker. Summarize the following phone call conversation.
    
    --- TRANSCRIPT ---
    {history_text}
    ------------------
    
    Please provide:
    1. The main goal of the caller.
    2. The outcome (Was an appointment booked? When?).
    3. Any follow-up actions required.
    """
    
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Failed to summarize: {e}"