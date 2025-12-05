from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from typing import List
from crewai import LLM
from pydantic import BaseModel
from agents.face_recognize import face_recognice
import os
from dotenv import load_dotenv
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from pathlib import Path

load_dotenv()

user_data = face_recognice()

user_profile = StringKnowledgeSource(
    content=f"User profile: Name: {user_data['Name']}, Age: {user_data['Age']}, Gender: {user_data['Gender']}, Allergies: {user_data['Allergies']}",
)

class PrescriptionOutput(BaseModel):
    Medicine: List[str]
    Dosage: str
    Treatment: str

@CrewBase
class SpeechCrew:
    """Medical consultation crew"""
    
    agents_config = 'config/speech_agent.yaml'
    tasks_config = 'config/speech_task.yaml'



    @agent
    def doctor(self) -> Agent:
        return Agent(
            config=self.agents_config["doctor"],
            verbose=False,
            max_iter=1,
            llm=LLM(model="gemini/gemini-2.0-flash", temperature=0.2),
            knowledge_sources=[user_profile],
        )

    @task
    def medical_consultation(self) -> Task:
        return Task(
            config=self.tasks_config["medical_consultation"],
            agent=self.doctor(),
            output_pydantic=PrescriptionOutput,
        )
    
    @crew
    def crew(self) -> Crew:

        os.environ["EMBEDDINGS_OLLAMA_MODEL_NAME"] = "mxbai-embed-large"
        project_root = Path(__file__).parent
        storage_dir = project_root / f"./medical_memory/{user_data['ID']}"
        os.environ["CREWAI_STORAGE_DIR"] = str(storage_dir)

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose = False,
            memory=True,
                embedder={
                    "provider": "ollama",
                    "config": {
                        "model": "mxbai-embed-large", 
                        "url": "http://localhost:11434/api/embeddings"
                    }
                }
        )

@CrewBase
class FileCrew:
    """This Crew Help To extract medical information from the file"""

    agents_config = 'config/file_agent.yaml'
    tasks_config = 'config/file_task.yaml'

    @agent
    def Medical_Data_Extractor(self) -> Agent:
        return Agent(
            config=self.agents_config["Medical_Data_Extractor"],
            verbose=False,
            llm=LLM(model="gemini/gemini-2.0-flash", temperature=0.1),
        )
    
    @task
    def medical_data(self) -> Task:
        return Task(
            config=self.tasks_config["medical_data"],
            agent=self.Medical_Data_Extractor(),
            output_pydantic=PrescriptionOutput,
        )
    
    @crew
    def crew(self) -> Crew:
        os.environ["CREWAI_STORAGE_DIR"] = f"./medical_memory/{user_data['ID']}"

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            memory=True,
            knowledge_sources=[user_profile],
            embedder={
                    "provider": "ollama",
                    "config": {
                        "model": "mxbai-embed-large", 
                        "url": "http://localhost:11434/api/embeddings"
                    }
            }
        )
    
