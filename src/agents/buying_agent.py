from browser_use import Agent, Browser, ChatGoogle, Tools
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import asyncio
from typing import List
load_dotenv()

tools = Tools()

class BestDeal(BaseModel):
    best_site: str
    product_name: str
    quantity: str
    price: float
    reason: str

async def run_comparison(medicine: List[str]):
    llm = ChatGoogle(model="gemini-flash-latest")

    browser = Browser(
        executable_path="C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
        user_data_dir=r"C:\agent_profile_parent",
        profile_directory="Default",                  
        permissions=["geolocation"],
        minimum_wait_page_load_time=2.0,  
        wait_for_network_idle_page_load_time=2.0,
        wait_between_actions=1.0,
    )

    task = f"""
    Give me the data of the each of the medicine in the given List: {medicine}, on these sites: 1mg and Apollo Pharmacy.

    1. Use go_to_url action to go to https://www.apollopharmacy.in/search-medicines?source=/
       - Search for each of the medicine in the list: {medicine}
       - After search on each medicine in the list Extract product name, price, delivery info, quantity, and offers of the first product after the search . 
    2. Use go_to_url action to go to https://www.1mg.com
       - Search for each of the medicine in the list: {medicine}
       - After search on each medicine in the list Extract product name, price, delivery info, quantity, and offers of the first product after the search . 
    3. Compare and return ONLY THE BEST SITE with:
       - best_site: which site name has better deal
       - product_name: product name of each medicine
       - quantity: quantity
       - price: total price
       - reason: why this is the best deal (consider price, delivery time, offers)

    4. use go_to_url acrion to visit the site which has better deal 
     - search for the {medicine} 
     - click on the first product get after the search
     - Click on the Add to cart element to add it in the cart
    """

    agent = Agent(
        task=task,
        llm=llm,
        browser=browser,
        tools=tools,
        output_model_schema=BestDeal
    )

    history = await agent.run()
    result = history.final_result()

    if not result:
        print("No result")
        return None

    best = history.structured_output
    return best
