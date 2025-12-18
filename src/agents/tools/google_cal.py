import os
import os.path
from typing import Type, Optional
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/calendar']

class CalendarEventInput(BaseModel):
    user_id: str = Field(..., description="Unique ID for the user (e.g. 'mom', 'dad'). This selects the specific 'token_{user_id}.json' file.")
    summary: str = Field(..., description="Title of the medicine/event (e.g., 'Take Metformin').")
    start_time: str = Field(..., description="Start time in ISO 8601 format (e.g., '2025-12-08T10:00:00').")
    end_time: str = Field(..., description="End time in ISO 8601 format (e.g., '2025-12-08T10:15:00').")
    description: Optional[str] = Field(None, description="Dosage instructions or notes.")
    location: Optional[str] = Field(None, description="Location (optional).")

class GoogleCalendarTool(BaseTool):
    name: str = "Schedule Medicine Event"
    description: str = "Creates a Google Calendar event for a specific family member using their unique account."
    args_schema: Type[BaseModel] = CalendarEventInput

    def _run(self, user_id: str, summary: str, start_time: str, end_time: str, description: str = None, location: str = None) -> str:
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        CREDENTIALS_PATH = os.path.join(BASE_DIR, 'credentials.json')
        TOKEN_PATH = os.path.join(BASE_DIR, f'token_{user_id}.json')

        creds = None

        try:

            if os.path.exists(TOKEN_PATH):
                creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    print(f"🔄 Refreshing token for user: {user_id}")
                    creds.refresh(Request())
                else:
                    print(f"👤 Initiating new login for user: {user_id}")
                    
                    if not os.path.exists(CREDENTIALS_PATH):
                        return f"❌ Error: 'credentials.json' not found at {CREDENTIALS_PATH}. Please make sure it is in the 'tools' folder."
                    
                    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
                    creds = flow.run_local_server(port=0)
                
                with open(TOKEN_PATH, 'w') as token:
                    token.write(creds.to_json())

            service = build('calendar', 'v3', credentials=creds)

            event_body = {
                'summary': summary,
                'location': location,
                'description': description,
                'start': {
                    'dateTime': start_time,
                    'timeZone': 'Asia/Kolkata', 
                },
                'end': {
                    'dateTime': end_time,
                    'timeZone': 'Asia/Kolkata',
                },
            }

            event_result = service.events().insert(calendarId='primary', body=event_body).execute()
            
            return f"✅ Success: Scheduled '{summary}' for ID '{user_id}'. Link: {event_result.get('htmlLink')}"

        except Exception as e:
            return f"❌ Error scheduling event for '{user_id}': {str(e)}"