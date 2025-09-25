import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.xai import xAI
from textwrap import dedent
from agno.tools.yfinance import YFinanceTools
from agno.tools.googlesearch import GoogleSearchTools

load_dotenv()
XAI_API_KEY = os.getenv("XAI_API_KEY")

# Definition: Number of calls to external APIs during task completion

def get_date_time():
    # return 'The current date and time is 2025-09-04 10:00:00'
    raise Exception("Simulated failure for testing")

agent = Agent(
    name="Helpful Assistant",
    model=xAI(id="grok-3", api_key=XAI_API_KEY),
    # model=xAI(id='grok-3-mini'),
    instructions=dedent("""
        You are a helpful assistant.
        - If the user explicitly asks for the current date or time, call 'get_date_time'.
        - If the user asks for up-to-date information (e.g., stock prices, news), call the 'google_search' or 'yfinance' tool.
        - Always include the tool's result in your final answer.
        - Do not call 'get_date_time' unless the query is specifically about the date or time.
    """),
    tools=[GoogleSearchTools(), YFinanceTools(), get_date_time],
    show_tool_calls=True,
)

# response = agent.run("what's the latest news in Finland today? In one sentence")
response = agent.run("What is the stock price of Honda? And what are the latest news headlines in Japan today?")
print('Response:', response.content)

external_tools = ("google_search", "get_current_stock_price")
# Count only external API calls
num_api_calls = sum(1 for tool in response.tools if tool.tool_name in external_tools)
print(f"Number of external API calls: {num_api_calls}")
