from textwrap import dedent
from agno.agent import Agent
from datetime import datetime
from agno.models.xai import xAI
from agno.tools.reasoning import ReasoningTools
from agno.tools.googlesearch import GoogleSearchTools
import os
from dotenv import load_dotenv
load_dotenv()
XAI_API_KEY = os.getenv("XAI_API_KEY")
# Cost per request
# Definition: Financial cost to complete an assigned task

def get_date_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

agent = Agent(
    name="Helpful Assistant",
    model=xAI(id='grok-3', api_key=XAI_API_KEY),
    # model=xAI(id='grok-3-mini'),
    instructions=dedent("""
        You are a helpful assistant.
        If the user explicitly asks for the current date or time, call 'get_date_time'.
        If the user asks for up-to-date information (e.g., stock prices, news), call the 'google_search' tool.
        Always include the tool's result in your final answer.
        Do not call 'get_date_time' unless the query is specifically about the date or time.
    """),
    tools=[GoogleSearchTools(), get_date_time],
    show_tool_calls=True,
)

response = agent.run("what's the current date and time? and what are the current trends in artificial intelligence and machine learning?")
# print('Response:', response.content)

# Sum tokens
total_input_tokens = sum(response.metrics['input_tokens'])
total_output_tokens = sum(response.metrics['output_tokens'])
total_tokens = sum(response.metrics['total_tokens'])

# Pricing per 1M tokens
input_price_per_m = 3 # $3 per million input tokens
output_price_per_m = 15 # $15 per million output tokens

# Calculate cost
input_cost = total_input_tokens / 1_000_000 * input_price_per_m
output_cost = total_output_tokens / 1_000_000 * output_price_per_m
total_cost = input_cost + output_cost

print("total tokens: ", total_tokens)
print("Input cost: $", input_cost)
print("Output cost: $", output_cost)
print("Total cost: $", total_cost)
