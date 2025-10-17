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

    address_info = {
        'street': 'IIITA Road',
        'city': 'Allahabad',
        'area':'jhalwa',
        'pincode': '211012',
        'phone': '9876543210',
        'landmark':'IIIT Allahabad'
    }

    browser = Browser(
        executable_path="C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
        permissions=["geolocation"],
        storage_state='./pharmacy_session.json',
        minimum_wait_page_load_time=2.0,  
        wait_for_network_idle_page_load_time=2.0,
        wait_between_actions=1.0  
    )

    task = f"""
    Give me the data of the each of the medicine in the given List: {medicine}, on these sites: 1mg and Apollo Pharmacy.

    Follow these steps:

    IMPORTANT - First Time Setup:
    if the delivery adrress is not similar to {address_info} then do the following but if it is similar to {address_info} skip these 4 step
    When visiting 1mg.com or apollopharmacy.in for the FIRST TIME:
    1. Look for location/delivery address prompt or icon (usually at top of page)

    2. Click on it and enter the delivery address using {address_info} appropriately according to what site asked for delivery location
    3. Save/confirm the delivery location

    4. Wait for the page to update with the new location

    If the site already has a saved delivery location, skip this step and proceed.

    1. Use go_to_url action to go to https://www.1mg.com
       - Search for each of the medicine in the list: {medicine}
       - After search on each medicine in the list Extract product name, price, delivery info, quantity, and offers of the first product that appears after the search

    2. Use go_to_url action to go to https://www.apollopharmacy.in/search-medicines?source=/
       - Search for each of the medicine in the list: {medicine}
       - After search on each medicine in the list Extract product name, price, delivery info, quantity, and offers of the first product that appears after the search

    3. Compare and return ONLY THE BEST SITE with:
       - best_site: which site name has better deal
       - product_name: product name of each medicine
       - quantity: quantity
       - price: total price
       - reason: why this is the best deal (consider price, delivery time, offers)
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

asyncio.run(run_comparison(["Paracetamol 500mg","Dolomine 650 mg Tablet"]))