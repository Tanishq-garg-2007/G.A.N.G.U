from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import requests
from urllib.parse import urlencode

class MyCustomToolInput(BaseModel):
    """Input schema for Hospital Finder Tool."""
    city: str = Field(..., description="City name, e.g., Aurangabad")
    state: str = Field(..., description="State name, e.g., Maharashtra")
    country: str = Field(..., description="Country name, e.g., India")
    limit: int = Field(10, description="Number of hospitals to return (default 10)")

class MyCustomTool(BaseTool):
    name: str = "Hospital Finder Tool"
    description: str = "Fetches top hospitals in a given city using Nominatim OpenStreetMap API."
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, city: str, state: str, country: str, limit: int = 10) -> str:
        """Fetch hospitals using Nominatim Search API."""
        try:
            params = {
                "city": city,
                "state": state,
                "country": country,
                "amenity": "hospital",
                "format": "json",
                "limit": limit
            }
            url = f"https://nominatim.openstreetmap.org/search?{urlencode(params)}"
            headers = {"User-Agent": "NearbyHospitalsApp/1.0"}
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code != 200:
                return f"Error fetching hospitals: {response.status_code}"

            data = response.json()
            if not data:
                return "No hospitals found."

            formatted = "\n".join(
                [f"{i+1}. {item.get('display_name')} (lat: {item.get('lat')}, lon: {item.get('lon')})" 
                 for i, item in enumerate(data)]
            )
            return formatted

        except Exception as e:
            return f"Error occurred: {str(e)}"