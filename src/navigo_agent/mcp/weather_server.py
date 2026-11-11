"""Custom Weather MCP Server — wraps OpenWeatherMap API as MCP tools.

Launched as a stdio subprocess by the MCP client (client.py).
Exposes two tools: get_current_weather and get_forecast.

Requires: OPENWEATHER_API_KEY in .env (free tier: https://openweathermap.org/api)
"""

import os
from collections import Counter

import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("Weather MCP Server")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
REQUEST_TIMEOUT_SECONDS = 10  # never hang the agent on a slow weather API


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
        timeout=REQUEST_TIMEOUT_SECONDS,
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


def summarise_forecast(entries: list[dict]) -> list[dict]:
    """Collapse 3-hourly forecast entries into one row per day.

    Each row: date, min/max temperature (C) and the most frequent condition.
    """
    days: dict[str, dict] = {}
    for item in entries:
        day = str(item.get("dt_txt", ""))[:10]
        main = item.get("main") or {}
        if not day or "temp" not in main:
            continue
        bucket = days.setdefault(day, {"temps": [], "conditions": Counter()})
        bucket["temps"].append(main.get("temp_min", main["temp"]))
        bucket["temps"].append(main.get("temp_max", main["temp"]))
        weather = item.get("weather") or [{}]
        bucket["conditions"][weather[0].get("description", "unknown")] += 1

    return [
        {
            "date": day,
            "temp_min_c": round(min(bucket["temps"]), 1),
            "temp_max_c": round(max(bucket["temps"]), 1),
            "condition": bucket["conditions"].most_common(1)[0][0],
        }
        for day, bucket in sorted(days.items())
    ]


@mcp.tool()
def get_forecast(city: str) -> dict:
    """Fetch the 5-day forecast for a city, summarised per day.

    Returns one entry per day: date, min/max temperature (C), dominant conditions.
    """
    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }

    response = requests.get(
        "https://api.openweathermap.org/data/2.5/forecast",
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    data = response.json()

    # Unknown city / bad key: pass the API error through instead of KeyError
    if response.status_code != 200 or "list" not in data:
        return data

    # The API returns ~40 three-hourly entries. A per-day summary covers the
    # whole 5 days in a handful of lines instead of only the next ~15 hours.
    return {
        "city": city,
        "daily": summarise_forecast(data["list"]),
    }


if __name__ == "__main__":
    mcp.run()
