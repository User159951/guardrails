from textwrap import dedent
from datetime import datetime
from agno.agent import Agent
from agno.models.xai import xAI
from agno.tools.googlesearch import GoogleSearchTools
from agno.eval.accuracy import AccuracyEval, AccuracyResult
from typing import Optional
import os
import dotenv

dotenv.load_dotenv()

XAI_API_KEY = os.getenv("XAI_API_KEY")
# Output Drift

# ---------- TOOLS ----------
def get_date_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ---------- AGNO AGENT ----------
agent = Agent(
    name="Climate Change Research Assistant",
    # model=MistralChat(id="mistral-large-latest"),
    model=xAI(id='grok-3-mini', api_key=XAI_API_KEY),
    instructions=dedent("""
        You are an assistant specialized in climate change research and environmental solutions in Africa.
        - If the user explicitly asks for the current date or time, call 'get_date_time'.
        - If the user asks for up-to-date info (e.g., news, research), call 'google_search'.
        - Always include the tool's result in your final answer.
        """),
    tools=[GoogleSearchTools(), get_date_time],
    show_tool_calls=True
)


evaluation = AccuracyEval(
    model=xAI(id="grok-3", api_key=XAI_API_KEY),
    agent=agent,
    input="What are the latest climate change solutions and technologies in Africa?",
    expected_output="Recent climate change solutions in Africa include renewable energy innovations, carbon capture technologies, sustainable agriculture practices, and green transportation systems",
    additional_guidelines="Check if the agent's response contains the same key information and style as the expected output.",
)

result: Optional[AccuracyResult] = evaluation.run(print_results=True)
