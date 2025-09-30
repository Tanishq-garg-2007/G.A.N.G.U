from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from browser_use import Browser, Agent, ChatGoogle
from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""
    argument: str = Field(..., description="The exact detail of the task assigned to the agent to do with proper instruction to navigate to the browser and search for desired product with urls of different platform mentioned in the task")

class MyCustomTool(BaseTool):
    name: str = "Gather Medicine Data"
    description: str = "This tool is used to gather medicine data like price, offers, availability etc. from different pharmacy websites"
    args_schema: Type[BaseModel] = MyCustomToolInput
    
    def _run(self, argument: str) -> str:
        """Custom function that uses browser_use library"""
        return asyncio.run(self._async_run(argument))
    
    async def _async_run(self, argument: str) -> str:
        """Async implementation of the browser automation"""
        browser = None
        try:
            browser = Browser(
                executable_path='C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
                minimum_wait_page_load_time=2.0,
                wait_for_network_idle_page_load_time=1.0,
                headless=False
            )
            
            agent = Agent(
                task=argument,
                browser=browser,
                llm=ChatGoogle(model='gemini-flash-latest',api_key="AIzaSyD3cxOBBAjaPd9k-okQWLk_VW16AX1V5ZM")
            )
            
            result = await agent.run()
            print(result.urls())
            print(argument)
            print(f"Has errors: {result.has_errors()}")
            print(f"Number of steps: {result.number_of_steps()}")
            print(result.errors())
            extracted = result.extracted_content()
            print("Extracted content:", extracted)

            if result.is_done():
                final_result = result.final_result() 
                if final_result:
                    return str(final_result)
                else:
                    return "Task completed but no data extracted"
            else:
                extracted = result.extracted_content()
                if extracted:
                    return "\n".join(str(item) for item in extracted if item)
                else:
                    return f"Task incomplete after {result.number_of_steps()} steps"
                    
        except Exception as e:
            return f"Error occurred during browser automation: {str(e)}"
        
        finally:
            if browser:
                try:
                    await browser.close()
                except:
                    pass