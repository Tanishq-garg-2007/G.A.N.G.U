import os, json, asyncio, base64, time, threading, subprocess, requests
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.responses import PlainTextResponse
from twilio.rest import Client
import uvicorn
# Removed: from pyngrok import ngrok (We are using subprocess now)
from agents.Calling_Ai import decide_reply, summarize_conversation
from agents.speech_to_text import connect_assemblyai, send_audio
from agents.text_to_speech import text_to_speech

load_dotenv()

app = FastAPI()

client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))

CURRENT_CALL_DATA = {
    "disease": "General Checkup",
    "doctor": "Any Doctor",
    "slot": "Any Available Time"
}

# --- New Function to Start Ngrok via CMD/Subprocess ---
def start_ngrok_process(port=8000):
    """
    Starts ngrok via system command and retrieves the public URL.
    """
    try:
        # 1. Kill any existing ngrok processes (optional, prevents conflicts)
        subprocess.run("taskkill /IM ngrok.exe /F", shell=True, stderr=subprocess.DEVNULL)
        
        # 2. Start ngrok in a separate background process
        # This is equivalent to running "ngrok http 8000" in CMD
        ngrok_process = subprocess.Popen(
            ["ngrok", "http", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"Ngrok process started (PID: {ngrok_process.pid}). Waiting for tunnel...")

        # 3. Wait and fetch the public URL from ngrok's local API
        time.sleep(3)  # Give ngrok a moment to start
        max_retries = 5
        for _ in range(max_retries):
            try:
                response = requests.get("http://127.0.0.1:4040/api/tunnels")
                data = response.json()
                public_url = data['tunnels'][0]['public_url']
                # Ensure we use the https version
                if public_url.startswith("http://"):
                    public_url = public_url.replace("http://", "https://")
                return public_url
            except Exception:
                time.sleep(1)
        
        raise Exception("Could not fetch Ngrok URL. Is ngrok installed and in PATH?")

    except FileNotFoundError:
        print("Error: 'ngrok' command not found. Please ensure ngrok is installed and in your system PATH.")
        return None

@app.post("/call")
def make_call(to_number: str):
    ngrok_url = os.environ.get("NGROK_URL")
    if not ngrok_url: return {"error": "Ngrok URL missing"}
    
    print(f"Calling {to_number}...")
    
    call = client.calls.create(
        to=to_number,
        from_=os.getenv("TWILIO_PHONE_NUMBER"),
        url=f"{ngrok_url}/twiml"
    )
    return {"call_sid": call.sid}

@app.post("/twiml", response_class=PlainTextResponse)
async def twiml_webhook():
    ngrok_url = os.environ.get("NGROK_URL")
    return f"""
    <Response>
      <Say voice="Polly.Joanna">Hi I want to book An Appointement</Say>
      <Connect>
        <Stream url="wss://{ngrok_url.replace('https://', '')}/media" />
      </Connect>
    </Response>
    """.strip()

@app.websocket("/media")
async def media_stream(websocket: WebSocket):
    await websocket.accept()
    print("Twilio stream connected")

    ws_assemblyai = await connect_assemblyai()
    stream_sid = None

    history = [] 

    async def assemblyai_listener():
        nonlocal stream_sid
        nonlocal history 
        
        while True:
            try:
                msg = await ws_assemblyai.recv()
                data = json.loads(msg)
                
                if data.get("type") == "Turn" and data.get("end_of_turn"):
                    transcript = data.get("transcript")
                    if transcript:
                        print(f"User said: {transcript}")
                        
                        history.append(f"User: {transcript}")

                        reply = decide_reply(transcript, history,CURRENT_CALL_DATA["disease"],CURRENT_CALL_DATA["doctor"],CURRENT_CALL_DATA["slot"])
                        
                        print(f"AI reply: {reply}")
                        
                        history.append(f"AI: {reply}")

                        if(len(history)>20):
                            history = history[-15]

                        audio_bytes = text_to_speech(reply)
                        if audio_bytes and stream_sid:
                            payload = base64.b64encode(audio_bytes).decode("utf-8")
                            await websocket.send_text(json.dumps({
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {"payload": payload}
                            }))
                            print("Audio sent to phone")
                        
            except Exception as e:
                print(f"Error in listener: {e}")
                break
    
    listener_task = asyncio.create_task(assemblyai_listener())
    audio_buffer = bytearray()
    MIN_CHUNK_SIZE = 1000 

    try:
        async for message in websocket.iter_text():
            msg = json.loads(message)
            if msg.get("event") == "media":
                chunk = base64.b64decode(msg["media"]["payload"])
                audio_buffer.extend(chunk)
                if len(audio_buffer) >= MIN_CHUNK_SIZE:
                    await send_audio(ws_assemblyai, audio_buffer)
                    audio_buffer.clear()
            elif msg.get("event") == "start":
                stream_sid = msg["start"]["streamSid"]
                print(f"Stream started: {stream_sid}")
            elif msg.get("event") == "stop":
                print("Stream stopped")
                break
    finally:
        summary = summarize_conversation(history)
        print(summary)
        
        listener_task.cancel()
        await ws_assemblyai.close()
        await websocket.close()

def start_call_sequence(number):
    print("Waiting 5 seconds for server and ngrok to stabilize...")
    time.sleep(5) # Increased wait time slightly

    print(f"Triggering auto-call to {number}...")
    try:
        ngrok_url = os.environ.get("NGROK_URL")
        if not ngrok_url:
            print("Error: Ngrok URL not found, cannot initiate call.")
            return

        call = client.calls.create(
            to=number,
            from_=os.getenv("TWILIO_PHONE_NUMBER"),
            url=f"{ngrok_url}/twiml"
        )
        print(f"Call Initiated! SID: {call.sid}")
    except Exception as e:
        print(f"Failed to trigger call: {e}")

def start_call(number, disease, doctor_name, Availablity):
    port = 8000

    CURRENT_CALL_DATA["disease"] = disease
    CURRENT_CALL_DATA["doctor"] = doctor_name
    CURRENT_CALL_DATA["slot"] = Availablity

    # --- New Logic: Start Ngrok via CMD ---
    public_url = start_ngrok_process(port)
    
    if public_url:
        print(f"Ngrok Tunnel Active: {public_url}")
        os.environ["NGROK_URL"] = public_url
        
        # Start call sequence in a separate thread
        threading.Thread(target=start_call_sequence, args=(number,), daemon=True).start()

        # Start FastAPI server
        uvicorn.run(app, host="127.0.0.1", port=port)
    else:
        print("Failed to start Ngrok. Exiting...")