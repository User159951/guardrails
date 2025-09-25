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

### Latency per request / Time to completion ###
print("---" * 5, "Collected Metrics", "---" * 5)
result = agent.run_response.metrics
pprint(result)
pprint(f"Time to completion: {sum(result.get('time')):.1f} seconds")
print("---" * 20)
