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
class MedicalCrew: 
    """Medical consultation and price comparison crew"""
    
    def __init__(self):
        self.price_comparison_tool = MyCustomTool()

    @agent
    def doctor(self) -> Agent:
        return Agent(
            config=self.agents_config['doctor'],
            verbose=True,
            max_iter=1,
            llm=LLM(
                model="gemini/gemini-2.0-flash",
                temperature=0.2,
            ),
        )

    @agent
    def data_collector(self) -> Agent:
        return Agent(
            config=self.agents_config['data_collector'],
            tools=[self.price_comparison_tool], 
            verbose=True,
            max_iter=1,
            llm=LLM(
                model="gemini/gemini-2.0-flash",
                temperature=0.2,
            ),
        )

    @task
    def medical_consultation(self) -> Task:
        return Task(
            config=self.tasks_config['medical_consultation'],
            agent=self.doctor(),
            output_pydantic=PrescriptionOutput,
        )

    @task
    def gathering_data(self) -> Task:
        return Task(
            config=self.tasks_config['gathering_data'],
            agent=self.data_collector(),
            context=[self.medical_consultation()],
            output_file="report.md",
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Medical crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )