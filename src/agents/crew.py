from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from typing import Type, Optional, List
from crewai import LLM
from pydantic import BaseModel, Field
from agents.face_recognize import face_recognice, get_user_knowledge_profile
import os
from dotenv import load_dotenv
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from pathlib import Path
from .tools.custom_tool import MyCustomTool
from .tools.google_cal import GoogleCalendarTool

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class PrescriptionOutput(BaseModel):
    user_id: str = Field(..., description="Unique ID for the user (e.g. 'mom', 'dad'). This selects the specific 'token_{user_id}.json' file.")
    name: str = Field(..., description="The medicine name that should be taken")
    start_time: str = Field(..., description="Start time in ISO 8601 format (e.g., '2025-12-08T10:00:00').")
    end_time: str = Field(..., description="End time in ISO 8601 format (e.g., '2025-12-08T10:15:00').")
    description: str = Field(None, description="Dosage instructions or notes.")

@CrewBase
class SpeechCrew:
    """Medical consultation crew"""
    
    agents_config = 'config/speech_agent.yaml'
    tasks_config = 'config/speech_task.yaml'

    @agent
    def doctor(self) -> Agent:
        return Agent(
            config=self.agents_config["doctor"],
            verbose=True,
            max_iter=1,
            memory=True,
            llm=LLM(model="gemini/gemini-2.5-flash", temperature=0.2,api_key=GEMINI_API_KEY),
        )
    
    @agent
    def care_coordinator(self) -> Agent:
        return Agent(
            config=self.agents_config["care_coordinator"],
            verbose=True,
            max_iter=1,
            memory=False,
            llm=LLM(model="gemini/gemini-2.5-flash", temperature=0.2,api_key=GEMINI_API_KEY),
            tools=[GoogleCalendarTool()]
        ) 

    @task
    def medical_consultation(self) -> Task:
        return Task(
            config=self.tasks_config["medical_consultation"],
            agent=self.doctor(),
            output_pydantic=PrescriptionOutput,
        )

    @task
    def care_task(self) -> Task:
        return Task(
            config=self.tasks_config["care_task"],
            agent=self.care_coordinator(),
            context = [self.medical_consultation()],
        )
       
    @crew
    def crew(self) -> Crew:
        user_data, profile_file = get_user_knowledge_profile()

        os.environ["EMBEDDINGS_OLLAMA_MODEL_NAME"] = "mxbai-embed-large"
        project_root = Path(__file__).resolve().parents[2]
        storage_dir = project_root / f"./medical_memory/{user_data['ID']}"
        os.environ["CREWAI_STORAGE_DIR"] = str(storage_dir)
        profile_file = f"user_{user_data['ID']}/profile.txt"
          
        knowledge_source = TextFileKnowledgeSource(
            file_paths=[str(profile_file)]
        )

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose = False,
            knowledge_sources=[knowledge_source],
                embedder={
                    "provider": "ollama",
                    "config": {
                        "model": "mxbai-embed-large", 
                        "url": "http://localhost:11434/api/embeddings"
                    }
                },
            process=Process.sequential,
            max_rpm = 3
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
            verbose=True,
            max_rpm = 3,
            llm=LLM(model="gemini/gemini-2.5-flash", temperature=0.1,api_key=GEMINI_API_KEY),
        )
    
    @agent
    def care_coordinator(self) -> Agent:
        return Agent(
            config=self.agents_config["care_coordinator"],
            verbose=True,
            max_iter=1,
            memory=False,
            max_rpm = 3,
            llm=LLM(model="gemini/gemini-2.5-flash", temperature=0.2,api_key=GEMINI_API_KEY),
            tools=[GoogleCalendarTool()]
        ) 
    
    @task
    def medical_data(self) -> Task:
        return Task(
            config=self.tasks_config["medical_data"],
            agent=self.Medical_Data_Extractor(),
            output_pydantic=PrescriptionOutput,
        )
    
    @task
    def care_task(self) -> Task:
        return Task(
            config=self.tasks_config["care_task"],
            agent=self.care_coordinator(),
            context = [self.medical_data()],
        )
       
    
    @crew
    def crew(self) -> Crew:
        user_data, profile_file = get_user_knowledge_profile()

        os.environ["EMBEDDINGS_OLLAMA_MODEL_NAME"] = "mxbai-embed-large"
        project_root = Path(__file__).resolve().parents[2]
        storage_dir = project_root / f"./medical_memory/{user_data['ID']}"
        os.environ["CREWAI_STORAGE_DIR"] = str(storage_dir)
        profile_file = f"user_{user_data['ID']}/profile.txt"
          
        knowledge_source = TextFileKnowledgeSource(
            file_paths=[str(profile_file)]
        )

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose = False,
            max_rpm = 3,
            knowledge_sources=[knowledge_source],
                embedder={
                    "provider": "ollama",
                    "config": {
                        "model": "mxbai-embed-large", 
                        "url": "http://localhost:11434/api/embeddings"
                    }
                },
            process=Process.sequential,
        )
    
@CrewBase
class Nearbyhospitals():
    """Nearbyhospitals crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = 'config/hospital_agent.yaml'
    tasks_config = 'config/hospital_task.yaml'


    @agent
    def hospital_finder(self) -> Agent:
        return Agent(
            config=self.agents_config['hospital_finder'],  
            tools=[MyCustomTool()],
            llm=LLM(model="gemini/gemini-2.5-flash", temperature=0.2,api_key=GEMINI_API_KEY),
            verbose=True
        )
        
    @agent
    def hospital_reporter(self) -> Agent:
        return Agent(
        config=self.agents_config['hospital_reporter'],
        llm=LLM(model="gemini/gemini-2.5-flash", temperature=0.2,api_key=GEMINI_API_KEY),
        verbose=True
    )


    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],  
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'],  
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        user_data, profile_file = get_user_knowledge_profile()

        os.environ["EMBEDDINGS_OLLAMA_MODEL_NAME"] = "mxbai-embed-large"
        project_root = Path(__file__).resolve().parents[2]
        storage_dir = project_root / f"./medical_memory/{user_data['ID']}"
        os.environ["CREWAI_STORAGE_DIR"] = str(storage_dir)
        profile_file = f"user_{user_data['ID']}/profile.txt"
          
        knowledge_source = TextFileKnowledgeSource(
            file_paths=[str(profile_file)]
        )

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose = False,
            knowledge_sources=[knowledge_source],
                embedder={
                    "provider": "ollama",
                    "config": {
                        "model": "mxbai-embed-large", 
                        "url": "http://localhost:11434/api/embeddings"
                    }
                },
            process=Process.sequential,
            max_rpm = 3
        )
    
@CrewBase
class general_que():
    """Nearbyhospitals crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = 'config/general_que_agent.yaml'
    tasks_config = 'config/general_que_task.yaml'


    @agent
    def home_health_assistant(self) -> Agent:
        return Agent(
            config=self.agents_config['home_health_assistant'],  
            llm=LLM(model="gemini/gemini-2.5-flash", temperature=0.2,api_key=GEMINI_API_KEY),
            verbose=True
        )

    @task
    def general_health_assistance_task(self) -> Task:
        return Task(
            config=self.tasks_config['general_health_assistance_task'],  
        )

    @crew
    def crew(self) -> Crew:
        user_data, profile_file = get_user_knowledge_profile()

        os.environ["EMBEDDINGS_OLLAMA_MODEL_NAME"] = "mxbai-embed-large"
        project_root = Path(__file__).resolve().parents[2]
        storage_dir = project_root / f"./medical_memory/{user_data['ID']}"
        os.environ["CREWAI_STORAGE_DIR"] = str(storage_dir)
        profile_file = f"user_{user_data['ID']}/profile.txt"
          
        knowledge_source = TextFileKnowledgeSource(
            file_paths=[str(profile_file)]
        )

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose = False,
            knowledge_sources=[knowledge_source],
                embedder={
                    "provider": "ollama",
                    "config": {
                        "model": "mxbai-embed-large", 
                        "url": "http://localhost:11434/api/embeddings"
                    }
                },
            process=Process.sequential,
            max_rpm = 3
        )