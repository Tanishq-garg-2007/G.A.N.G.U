from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from typing import List
from crewai import LLM
from pydantic import BaseModel
from agents.tools.custom_tool import MyCustomTool

class PrescriptionOutput(BaseModel):
    Medicine: List[str]
    Dosage: str
    Treatment: str

@CrewBase
class SpeechCrew:
    """Medical consultation crew"""
    
    agents_config = 'config/speech_agent.yaml'
    tasks_config = 'config/speech_task.yaml'

    def __init__(self):
        self.price_comparison_tool = MyCustomTool()

    @agent
    def doctor(self) -> Agent:
        return Agent(
            config=self.agents_config["doctor"],
            verbose=True,
            max_iter=1,
            llm=LLM(model="gemini/gemini-2.0-flash", temperature=0.2),
        )

    @agent
    def data_collector(self) -> Agent:
        return Agent(
            config=self.agents_config["data_collector"],
            tools=[self.price_comparison_tool],
            verbose=True,
            max_iter=1,
            llm=LLM(model="gemini/gemini-2.0-flash", temperature=0.2),
        )

    @task
    def medical_consultation(self) -> Task:
        return Task(
            config=self.tasks_config["medical_consultation"],
            agent=self.doctor(),
            output_pydantic=PrescriptionOutput,
        )

    @task
    def gathering_data(self) -> Task:
        return Task(
            config=self.tasks_config["gathering_data"],
            agent=self.data_collector(),
            context=[self.medical_consultation()],
            output_file="report.md",
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

@CrewBase
class FileCrew:
    """This Crew Help To extract medical information from the file"""

    agents_config = 'config/file_agent.yaml'
    tasks_config = 'config/file_task.yaml'

    def __init__(self):
        self.price_comparison_tool = MyCustomTool()

    @agent
    def Medical_Data_Extractor(self) -> Agent:
        return Agent(
            config=self.agents_config["Medical_Data_Extractor"],
            verbose=True,
            llm=LLM(model="gemini/gemini-2.0-flash", temperature=0.1),
        )
    
    @agent
    def data_collector(self) -> Agent:
        return Agent(
            config=self.agents_config["data_collector"],
            tools=[self.price_comparison_tool],
            verbose=True,
            max_iter=1,
            llm=LLM(model="gemini/gemini-2.0-flash", temperature=0.2),
        )
    
    @task
    def medical_data(self) -> Task:
        return Task(
            config=self.tasks_config["medical_data"],
            agent=self.Medical_Data_Extractor(),
            output_pydantic=PrescriptionOutput,
        )
    
    @task
    def gathering_data(self) -> Task:
        return Task(
            config=self.tasks_config["gathering_data"],
            agent=self.data_collector(),
            context=[self.medical_data()],
            output_file="report.md",
        )
    
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )