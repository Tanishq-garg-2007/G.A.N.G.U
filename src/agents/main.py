#!/usr/bin/env python
import sys
from agents.crew import SpeechCrew, FileCrew
from agents.speech_to_text import speech_to_text
from agents.file_to_text import file_to_text
from agents.buying_agent import run_comparison
from datetime import datetime
import asyncio

def run(statement=None):
    """
    Run the crew.
    """
    choice = input("Enter Your Choice: ")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if choice == "1":
        statement = speech_to_text(duration=5)
        result = SpeechCrew().crew().kickoff(inputs={"statement": f"[{timestamp}] {statement}"})
        
        print(result)
        print(result.pydantic.Medicine)
        move_next = input("Do you want to extract medicine data from different platforms? (yes/no): ").strip().lower()
        if move_next == "yes":
            print("\nStarting price comparison...")
            asyncio.run(run_comparison(result.pydantic.Medicine))
            print("\nComparison complete! Check grocery_purchase_results/report.md")
        else:
            print("Have a nice day!")
        return result
    
    elif choice == "2":
        statement = file_to_text()
        result = FileCrew().crew().kickoff(inputs={"statement": f"[{timestamp}] {statement}"})
    else:
        return "Please Enter A Valid Choice"

    return result