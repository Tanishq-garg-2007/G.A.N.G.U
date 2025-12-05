import os
import requests
from dotenv import load_dotenv

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

VOICE_MODEL = "aura-asteria-en"

def text_to_speech(text):
    """
    Generates audio using Deepgram Aura (Free Tier friendly).
    Requesting 'mulaw' at 8000Hz for direct Twilio compatibility.
    """
    url = f"https://api.deepgram.com/v1/speak?model={VOICE_MODEL}&encoding=mulaw&sample_rate=8000&container=none"
    
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {"text": text}

    try:
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            return response.content
        else:
            print(f" Deepgram TTS Error {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        print(f" TTS Exception: {e}")
        return None
