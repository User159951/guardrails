from textwrap import dedent
from agno.agent import Agent
from datetime import datetime
from agno.models.xai import xAI
from agno.models.mistral import MistralChat
from agno.tools.googlesearch import GoogleSearchTools
import os
# Jailbrake detection
import dotenv

dotenv.load_dotenv()

XAI_API_KEY = os.getenv("XAI_API_KEY")  

# ---------- TOOLS ----------
def get_date_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ---------- AGNO AGENT ----------
agent = Agent(
    name="Helpful Assistant",
    # model=MistralChat(id="mistral-large-latest"),
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
# prompt = "What are the latest advancements in healthcare technology in Europe?"

JAILBREAK_KEYWORDS = [
    "ignore instructions",
    "pretend you are",
    "bypass restrictions",
    "disregard rules",
    "act as someone else",
    "break your safety",
    "circumvent limitations",
    "give me answers you normally wouldn't",
    "tell me secrets",
]

def detect_jailbreak(prompt):
    prompt_lower = prompt.lower()
    return any(kw in prompt_lower for kw in JAILBREAK_KEYWORDS)

while True:
    prompt = input("Enter a prompt: ")
    if prompt == "exit":
        break
    
    if detect_jailbreak(prompt):
        print("Jailbreak detected!")
        quit()
    response = agent.run(prompt)
    print('Response:', response.content)
