"""Custom Weather MCP Server — wraps OpenWeatherMap API as MCP tools.

Launched as a stdio subprocess by the MCP client (client.py).
Exposes two tools: get_current_weather and get_forecast.

Requires: OPENWEATHER_API_KEY in .env (free tier: https://openweathermap.org/api)
"""

import os

import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("Weather MCP Server")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")


@mcp.tool()
def get_current_weather(city: str) -> dict:
    """Fetch current weather conditions for a city.

    Returns temperature (C), feels-like, humidity, wind speed, and conditions.
    """
    response = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={
            "q": city,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric",
        },
    )
    data = response.json()

    # API returns error payload with 200-like status — pass through raw
    if response.status_code != 200:
        return data

    return {
        "city": data["name"],
        "temperature_c": data["main"]["temp"],
        "feels_like_c": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "condition": data["weather"][0]["description"],
        "wind_speed": data["wind"]["speed"],
    }


@mcp.tool()
def get_forecast(city: str) -> dict:
    """Fetch 5-day/3-hour forecast for a city.

    Returns the first 5 forecast intervals (covers ~15 hours).
    """
    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }

    response = requests.get(
        "https://api.openweathermap.org/data/2.5/forecast",
        params=params,
    )
    data = response.json()

    # Return first 5 entries — enough for a practical travel forecast
    # without overwhelming the LLM context window
    forecast = []
    for item in data["list"][:5]:
        forecast.append({
            "datetime": item["dt_txt"],
            "temperature": item["main"]["temp"],
            "weather": item["weather"][0]["description"],
        })

    return {
        "city": city,
        "forecast": forecast,
    }


if __name__ == "__main__":
    mcp.run()
