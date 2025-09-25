import os
import asyncio
from textwrap import dedent
from datetime import datetime
from agno.agent import Agent
from agno.models.xai import xAI
from agno.models.mistral import MistralChat
from agno.tools.reasoning import ReasoningTools
from agno.tools.googlesearch import GoogleSearchTools
import os
import dotenv

dotenv.load_dotenv()

XAI_API_KEY = os.getenv("XAI_API_KEY")
# Input Drift
# XL to chat, XK to generate

# Tools
def get_date_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# AGNO AGENT
agent = Agent(
    name="Healthcare Tech Assistant",
    #model=MistralChat(id="mistral-large-latest")
    model=xAI(id='grok-3-mini', api_key=XAI_API_KEY),
    instructions=dedent("""
        You are an assistant specialized in new healthcare technologies in Europe.
        If the user explicitly asks for the current date or time, call 'get_date_time'.
        If the user asks for up-to-date info (e.g., news, research), call 'google_search'.
        Always include the tool's result in your final answer.
    """),
    tools=[GoogleSearchTools(), get_date_time],
    show_tool_calls=True
)

# ----- JUDGE AGENT -----
input_drift_agent = Agent(
    name="Input Drift Judge Agent",
    #model=MistralChat(id="mistral-large-latest")
    model=xAI(id='grok-3-mini', api_key=XAI_API_KEY),
    instructions=dedent("""
        You are the Input Drift Judge Agent.
        
        Determine whether a new user prompt is 'in-domain' or 'out-of-domain' compared to baseline prompts focused on new healthcare technologies in Europe.
        
        You will be given:
        - A list of baseline prompts (examples of valid prompts)
        - A new user prompt
        
        Decide whether the new prompt belongs to the same domain as the baseline prompts.
        
        In-domain criteria:
        - Focused on healthcare technology in Europe
        - Related to medical devices, AI in healthcare, telemedicine, hospitals, biotech, etc.
        
        Out-of-domain criteria:
        - Topics unrelated to healthcare technology in Europe
        - Healthcare topics outside Europe
        
        Respond in this JSON format:
        {
          "is_drift": true/false,
          "reason": "Short explanation of why it is in-domain or out-of-domain"
        }
        
        If uncertain, conservatively assume it is out-of-domain.
        Only judge domain relevance; ignore spelling or grammar issues.
    """),
    tools=[ReasoningTools(add_instructions=True)],
    show_tool_calls=True
)

baseline_prompts = [
"What are the newest digital health innovations in European clinics?",
"Which European countries are leading in medical AI research?",
"How are European hospitals implementing smart monitoring systems?",
"What are the latest European regulations for medical device cybersecurity?",
"Which European biotech companies are developing breakthrough therapies?"
]

# ---------- RUN AGENT ----------

while True:
    prompt = input('Enter your prompt: ')
    judge_prompt = f"""
Here are example prompts about new healthcare technologies in Europe:
{baseline_prompts}

Does this new prompt fit the same domain?:
{prompt}
"""

    judge_response = input_drift_agent.run(judge_prompt)
    print('-----' * 5, 'Judge Response', '-----' * 5)
    print(judge_response.content)
    print('-----' * 10)
    agent_response = agent.run(prompt)
    print('-----' * 5, 'Agent Response', '-----' * 5)
    print(agent_response.content)
    # DEL to chat, WK to generate
    # ---------- TEST PROMPTS ----------
    # In-domain
    # What are the latest European developments in precision medicine?
    # Out-of-domain
    # What are the best restaurants in Paris?