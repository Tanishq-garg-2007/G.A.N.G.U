import sounddevice as sd
from scipy.io.wavfile import write
from dotenv import load_dotenv
import os
import assemblyai as aai

load_dotenv()
API_KEY = os.getenv("ASSEMBLY_API_KEY")
aai.settings.api_key = API_KEY

# ===== 1. Record audio from microphone =====
def record_audio(filename="speech.wav", duration=10, fs=16000):
    print(f"Speak now for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, recording)
    print(f"Saved audio to {filename}")
    return filename

# ===== 2. Transcribe audio =====
def transcribe_audio(filename):
    config = aai.TranscriptionConfig(language_code="hi")  # Hindi/HinEnglish
    transcriber = aai.Transcriber()
    
    print("Starting transcription...")
    transcript = transcriber.transcribe(filename, config=config)
    
    return transcript.text  # Return only the text

# ===== 3. Combined function for external use =====
def speech_to_text(duration=5):
    filename = record_audio(duration=duration)
    text = transcribe_audio(filename)
    return text  # This can be imported and used in other files

