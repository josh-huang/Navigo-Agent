import os
import sys
import asyncio
import certifi
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AVIATION_STACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")

client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"
        },
        "aviationstack": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "aviationstack_mcp"],
            "env": {
                "AVIATIONSTACK_API_KEY": AVIATION_STACK_API_KEY
            }
        },
    }
)


async def get_all_tools():
    tools = await client.get_tools()
    for tool in tools:
        print(f"Tool Name: {tool.name}")