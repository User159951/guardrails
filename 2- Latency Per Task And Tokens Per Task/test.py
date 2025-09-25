import os
import dotenv
import agentops
from agno.agent import Agent
from agno.models.mistral import MistralChat
from agno.tools.googlesearch import GoogleSearchTools
from textwrap import dedent
from pprint import pprint

dotenv.load_dotenv()

# Initialize AgentOps for monitoring
AGENTOPS_API_KEY = 'eb542050-6531-4772-a1f9-8981299f2140'
agentops.init(
    api_key=AGENTOPS_API_KEY,
    default_tags=['agno'],
    log_level='ERROR'
)

mistral_api_key = os.getenv('MISTRAL_API_KEY')

# Create the agent
agent = Agent(
    name="Research Assistant",
    model=MistralChat(id='mistral-small-latest', api_key=mistral_api_key),
    instructions=dedent("""
        You are a research assistant specialized in technology and science.
        You can search the web for the latest information and provide detailed analysis.
        Always cite your sources when providing information.
    """),
    tools=[GoogleSearchTools()],
    show_tool_calls=True,
)

# Execute a different query
agent.print_response("""
What are the latest developments in artificial intelligence and machine learning? 
Provide a brief overview of the most significant recent advances.
""", stream=True)

### Latency per task ### Tokens usage per task ###
if agent.run_response.messages:
    for message in agent.run_response.messages:
        print('->', message.role)
        if message.role == "assistant":
            if message.content:
                print(f"Message: {message.content}")
            elif message.tool_calls:
                print(f"Tool calls: {message.tool_calls}")
        print("---" * 5, "Metrics", "---" * 5)
        pprint(message.metrics)
        print("---" * 20)
