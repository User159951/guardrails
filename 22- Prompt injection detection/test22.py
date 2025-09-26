import os
from textwrap import dedent
from datetime import datetime
from agno.agent import Agent
from agno.models.xai import xAI
from agno.tools.googlesearch import GoogleSearchTools
import os
import dotenv

dotenv.load_dotenv()

XAI_API_KEY = os.getenv("XAI_API_KEY")
# Prompt Injection Detection

# ---------- TOOLS ----------
def get_date_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------- AGNO AGENT ----------
agent = Agent(
    name="Healthcare Tech Assistant",
    # model=MistralChat(id="mistral-large-latest"),
    model=xAI(id='grok-3-mini', api_key=XAI_API_KEY),
    instructions=dedent("""
        You are an assistant specialized in new healthcare technologies in Europe.
        - If the user explicitly asks for the current date or time, call 'get_date_time'.
        - If the user asks for up-to-date info (e.g., news, research), call 'google_search'.
        - Always include the tool's result in your final answer.
        """),
    tools=[GoogleSearchTools(), get_date_time],
    show_tool_calls=True
)

# ---------- LLM-GUARD PROMPT INJECTION ----------

from llm_guard.input_scanners import PromptInjection
from llm_guard.input_scanners.prompt_injection import MatchType

scanner = PromptInjection(threshold=0.5, match_type=MatchType.FULL)

# TODO: Add other LLM to generate
while True:
    prompt = input("Enter your prompt: ")
    sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)
    if is_valid:
        print('✅ Passed to the agent')
        response = agent.run(prompt)
        print('Agent Response:', response.content)
    else:
        print('❌ Rejected')
