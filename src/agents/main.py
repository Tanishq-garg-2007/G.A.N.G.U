#!/usr/bin/env python
import sys
from agents.crew import MedicalCrew
from agents.speech_to_text import speech_to_text

def run(statement=None):
    """
    Run the crew.
    """
    statement = speech_to_text(duration=5) 
    print(f"Transcribed: {statement}")
    
    inputs = {
        "statement": statement
    }
    
    result = MedicalCrew().crew().kickoff(inputs=inputs)
    return result
