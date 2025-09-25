import os
import json
import asyncio
from textwrap import dedent
from agno.agent import Agent
from datetime import datetime
from agno.models.xai import xAI
from agno.models.mistral import MistralChat
from agno.tools.googlesearch import GoogleSearchTools
import dotenv
dotenv.load_dotenv()
XAI_API_KEY = os.getenv("XAI_API_KEY")
# Infinite loop detection

# ----- TOOLS -----
def get_date_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ----- AGNO AGENT -----
agent = Agent(
    name="Helpful Assistant",
    #model=MistralChat(id="mistral-large-latest")
    model=xAI(id="grok-3-mini", api_key=XAI_API_KEY),
    instructions=dedent("""
        You are a helpful assistant.
        If the user explicitly asks for the current date or time, call 'get_date_time'.
        If the user asks for up-to-date information (e.g., stock prices, news), call the 'google_search' tool.
        Always include the tool's result in your final answer.
        Do not call 'get_date_time' unless the query is specifically about the date or time.
    """),
    tools=[GoogleSearchTools(), get_date_time],
    show_tool_calls=True
)


# ---------- RUN AGENT ----------
prompt = "what are the current trends in artificial intelligence and machine learning?"
response = agent.run(prompt, stream=True, stream_intermediate_steps=True)

import asyncio
import time

timeout_seconds = 0.01
last_event_time = time.time()

for event in response:
    if event.event in ("ReasoningStep", "ToolCallStarted", "RunResponseContent"):
        last_event_time = time.time()

    if event.event in ("RunCompleted", "RunError"):
        break # agent finished normally

    # Check for infinite loop
    if time.time() - last_event_time > timeout_seconds:
        # Agent stuck, infinite loop suspected
        print("Potential infinite loop detected!")
        quit()

print('✅ All good!')
