import os
import asyncio
from textwrap import dedent
from datetime import datetime
from agno.agent import Agent
from agno.models.xai import xAI
from agno.models.mistral import MistralChat
from agno.tools.reasoning import ReasoningTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.googlesearch import GoogleSearchTools
import os
import dotenv

dotenv.load_dotenv()

XAI_API_KEY = os.getenv("XAI_API_KEY")
# Off Topic Detection

# ---------- TOOLS ----------
def get_date_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ---------- AGNO AGENT ----------
agent = Agent(
    name="AI Research Assistant",
    # model=xAI(id='grok-3-mini'),
    model=xAI(id='grok-3-mini', api_key=XAI_API_KEY),
    instructions=dedent("""
        You are an assistant specialized in artificial intelligence research and development.
        - If the user explicitly asks for the current date or time, call 'get_date_time'.
        - If the user asks for up-to-date info (e.g., news, research), call 'DuckDuckGoTools'.
        - Always include the tool's result in your final answer.
    """),
    tools=[DuckDuckGoTools(), get_date_time],
)

# ---------- OFF-TOPIC JUDGE AGENT ----------
offtopic_judge = Agent(
    name="OffTopic Judge",
    model=xAI(id='grok-3-mini', api_key=XAI_API_KEY),
    instructions=dedent("""
        You are an Off-Topic Judge.
        The main assistant is ONLY about *artificial intelligence research and development*.
        Your job is to check if the user's query is on-topic or off-topic.

        Rules:
        - If the query is about AI, machine learning, deep learning, neural networks, computer vision, NLP, robotics, or related technology research -> respond with EXACTLY "On-topic".
        - If the query is outside this scope (sports, travel, politics, cooking, entertainment, etc.) -> respond with EXACTLY "Off-topic".
        - Do not explain. Output only one of: "On-topic" or "Off-topic".
    """)
)

# ---------- RUN AGENTS ----------
while True:
    prompt = input('Enter your prompt: ')
    judge_response = offtopic_judge.run(prompt)
    if judge_response.content.lower() == 'on-topic':
        response = agent.run(prompt)
        print('Agent Response:', response.content)
    else:
        print(dedent("""
I'm here to assist only with artificial intelligence research and development.
Your request seems to be outside this scope. Could you please rephrase your question to stay on topic?
"""))

# What's the best vacation spot in Italy?
# What are the latest advances in transformer architectures?

