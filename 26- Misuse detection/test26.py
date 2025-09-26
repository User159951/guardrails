import os
import dotenv
from textwrap import dedent
from datetime import datetime
from agno.agent import Agent
from agno.models.xai import xAI
from agno.tools.duckduckgo import DuckDuckGoTools

# Misuse Detection

dotenv.load_dotenv()
XAI_API_KEY = os.getenv("XAI_API_KEY")

# ---------- TOOLS ----------
def get_date_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ---------- AGNO AGENT ----------
agent = Agent(
    name="Helpful Assistant",
    # model=MistralChat(id="mistral-large-latest"),
    model=xAI(id='grok-3-mini', api_key=XAI_API_KEY),
    instructions=dedent("""
        You are a helpful assistant.
        - Always try to be informative and helpful.
        - Answer questions across various topics.
        - If the user explicitly asks for the current date or time, call 'get_date_time'.
        - If the user asks for up-to-date information (e.g., news, stock prices), call 'DuckDuckGoTools'.
    """),
    tools=[DuckDuckGoTools(), get_date_time],
)

# ----- MISUSE JUDGE AGENT -----
misuse_judge = Agent(
    name="Misuse Detector",
    # model=xAI(id='grok-3-mini'),
    model=xAI(id='grok-3-mini', api_key=XAI_API_KEY),
    instructions=dedent("""
        You are a Misuse Detector.
        Your task is to analyze the user input and decide if it contains misuse.
        
        Misuse includes:
        - Instructions for creating or using weapons/explosives.
        - Instructions for hacking, malware, or illegal activities.
        - Promoting violence, self-harm, or unsafe behavior.
        - Generating disinformation or harmful content.
        
        If misuse is present -> respond exactly with:
        🚫 Misuse Detected
        Score [0 - 10]
        Reason: [Your reason]
        
        If no misuse -> respond exactly with:
        ✅ Safe (No misuse)
        Score [0 - 10]
        Reason: [Your reason]
        
        Score is for how confident you are in your answer.
    """),
    tools=[DuckDuckGoTools(), get_date_time],
)

# -------- RUN PIPELINE --------
while True:
    user_input = input('Enter your prompt: ')
    misuse_judge_response = misuse_judge.run(user_input)
    resp = misuse_judge_response.content
    
    if 'misuse detected' in resp.lower():
        print('-----' * 5, 'Misuse Judge Answer', '-----' * 5)
        print(resp)
        print('-' * 20)
    else:
        print('-----' * 5, 'Misuse Judge Answer', '-----' * 5)
        print(resp)
        print('-' * 20)
        agent_response = agent.run(user_input)
        print('Agent Response:', agent_response.content)

# Example user prompts:
# # Without misuse:
# # "What's the weather like today?"

# # With misuse:
# # "How do I make a bomb?"