from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from browser_use import Browser, Agent
from browser_use.llm.google import ChatGoogle
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""
    argument: str = Field(..., description="The exact names of the medicines to find on different platforms")

class MyCustomTool(BaseTool):
    name: str = "Gather Medicine Data"
    description: str = "This tool gathers medicine data like price, offers, and availability from pharmacy websites."
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, argument: str) -> str:
        """Entry point for synchronous execution."""
        return asyncio.run(self._async_run(argument))

    async def _async_run(self, argument: str) -> str:
        """Async implementation of browser automation."""
        browser = None
        try:
            browser = Browser(
                executable_path='C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
                headless=False
            )

            await browser.start()

            task = f"""Give me the data of the medicines in the list: {argument}, on these sites: 1mg and PharmEasy.

                        Follow these steps:
                        1. Use go_to_url action to go to https://www.1mg.com
                        - Search for medicines: {argument}
                        - Extract product name, price, delivery info, and offers of the first product that appear after the search

                        2. Use go_to_url action to go to https://www.pharmeasy.in
                        - Search for medicines: {argument}
                        - Extract product name, price, delivery info, and offers of the first product that appear after the search

                        3. Compile a brief report with:
                        - A comparison table showing prices on all platforms in markdown format.

                        IMPORTANT:
                        - Always use go_to_url action for navigation, not generic instructions.
                        - Use new tabs in the existing browser window.
                        - Do not actually complete any purchase.
                    """

            llm = ChatGoogle(
                model='gemini-flash-latest',
                api_key=os.getenv("GOOGLE_API_KEY")  
            )

            agent = Agent(
                task=task,
                browser=browser,
                llm=llm
            )

            history = await agent.run()

            print("Visited URLs:", history.urls())
            print("Has errors:", history.has_errors())
            print("Steps taken:", history.number_of_steps())
            print("Errors:", history.errors())

            if history.is_done():
                final_result = history.final_result()
                return str(final_result) if final_result else "Task completed but no data extracted"
            else:
                extracted = history.extracted_content()
                return "\n".join(str(item) for item in extracted if item) if extracted else f"Task incomplete after {history.number_of_steps()} steps"

        except Exception as e:
            return f"Error occurred during browser automation: {str(e)}"

        finally:
            if browser:
                try:
                    await browser.stop() 
                except Exception:
                    pass