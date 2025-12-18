#!/usr/bin/env python
import sys
from agents.crew import SpeechCrew, FileCrew, Nearbyhospitals ,general_que
from agents.speech_to_text import speech_to_text
from agents.file_to_text import file_to_text
from agents.buying_agent import run_comparison
from datetime import datetime
from agents.Calling import start_call
from agents.face_recognize import face_recognice

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
        move_next = input("Do you want to extract medicine data from different platforms? (yes/no): ").strip().lower()
        if move_next == "yes":
            print("\nStarting price comparison...")
            asyncio.run(run_comparison(result.pydantic.name))
            print("\nComparison complete! Check grocery_purchase_results/report.md")
        else:
            print("Have a nice day!")
        return result
    
    elif choice == "2":
        statement = file_to_text()
        result = FileCrew().crew().kickoff(inputs={"statement": f"[{timestamp}] {statement}"})

        print(result)

        move_next = input("Do you want to extract medicine data from different platforms? (yes/no): ").strip().lower()
        if move_next == "yes":
            print("\nStarting price comparison...")
            asyncio.run(run_comparison(result.pydantic.name))
            print("\nComparison complete! Check grocery_purchase_results/report.md")
        else:
            print("Have a nice day!")
        return result

    elif choice == "3":
        number = input("Please enter the number you want to call")
        disease = input("please enter the disease ")
        doctor_name = input("please enter the doctor name ")
        Availablity = input("Please tell your time slot ")

        start_call(number,disease,doctor_name,Availablity)

    elif choice == "4":
        user_profile = face_recognice()
        city = user_profile["City"]
        state = user_profile["State"]
        
        inputs = {
            "city" : city,
            "state" : state,
            "country" : "India",
            "limit" : 5
        }

        result = Nearbyhospitals().crew().kickoff(inputs=inputs)

        print("Hospital Found: ")
        if isinstance(result, list):
            for i, hospital in enumerate(result, start=1):
                print(f"{i}. {hospital}")
        else:
            print(result)

    elif choice == "5":
        question = speech_to_text(duration=5)

        result = general_que().crew().kickoff(inputs={"question" : question})
        print(result)
        
    else:
        return "Please Enter A Valid Choice"

    return result