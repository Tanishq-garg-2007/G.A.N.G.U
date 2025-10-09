#!/usr/bin/env python
import sys
from agents.crew import SpeechCrew, FileCrew
from agents.speech_to_text import speech_to_text
from agents.file_to_text import file_to_text

def run(statement=None):
    """
    Run the crew.
    """
    choice = input("Enter Your Choice: ")

    if choice == "1":
        statement = speech_to_text(duration=5)
        result = SpeechCrew().crew().kickoff(inputs={"statement": statement})
    elif choice == "2":
        statement = file_to_text()
        result = FileCrew().crew().kickoff(inputs={"statement": statement})
    else:
        return "Please Enter A Valid Choice"

    return result