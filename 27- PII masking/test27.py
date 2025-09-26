import re
import os
import dotenv
from textwrap import dedent
from datetime import datetime
from agno.agent import Agent
from agno.models.xai import xAI
from agno.tools.duckduckgo import DuckDuckGoTools

dotenv.load_dotenv()
XAI_API_KEY = os.getenv("XAI_API_KEY")

# PII (Personally Identifiable Information) Masking

# ---------- TOOLS
def get_date_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ---------- AGNO AGENT
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

# ---------- PII MASKING USING REGEX-BASED DETECTION ----------
def mask_pii(text):
    # Masks email addresses
    text = re.sub(r'[A-Za-z0-9.%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', '[EMAIL]', text)
    # Masks US SSN (Social Security Number)
    text = re.sub(r'\b\d{3}[-.\s]??\d{2}[-.\s]??\d{4}\b', '[SSN]', text)  # US SSN
    # Masks phone numbers (international format with optional '+' and varying length)
    text = re.sub(r'\+?\d{1,3}[-.\s]??\d{6,14}', '[PHONE]', text)
    return text

# Example usage and expected output for the mask_pii function:
# sample = "Contact me at john@example.com or +44 7911 123456"
# print(mask_pii(sample))
# Output: "Contact me at [EMAIL] or [PHONE]"
# 123-45-6789 (This line appears to be a comment showing an example SSN format)

# ---------- RUN AGENT ----------
# This section demonstrates a loop where user input is taken, PII masked, and then passed to an agent.
while True:
    user_input = input('Enter your prompt: ')
    print('Before PII Masking:', user_input)
    user_input = mask_pii(user_input)
    print('After PII Masking:', user_input)
    print('-'*20)  # Prints a separator line
    # Assuming 'agent' is an object with a 'run' method that takes the masked input
    response = agent.run(user_input)
    print('Agent Response:', response.content)
