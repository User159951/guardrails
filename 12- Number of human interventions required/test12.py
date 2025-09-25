from textwrap import dedent
from agno.agent import Agent
from rich.prompt import Prompt
from agno.models.xai import xAI
import os
from dotenv import load_dotenv

load_dotenv()
XAI_API_KEY = os.getenv("XAI_API_KEY")

def human_intervention(requested_info: str):
    input_text = Prompt.ask(requested_info)
    return input_text

agent = Agent(
    name="Professional Email Signature Generator",
    model=xAI(id="grok-3", api_key=XAI_API_KEY),
    description=dedent("""
    You are a helpful assistant that generates professional email signatures based on the user's input.
    """),
    instructions=dedent("""
    You should be given an input that includes full name, job title, company name, phone number, email address, and optional elements like website, LinkedIn profile, or social media handles
    Based on this input, generate a professional email signature that:
    Includes all essential contact information
    Maintains a clean, professional format
    Uses appropriate hierarchy (name, title, company, contact details)
    Follows email signature best practices
    Is concise but informative
    
    If any of the required input details are missing, call 'human_intervention(requested_info: str)', where requested_info is a question asking the user to provide all the missing information.
    """),
    tools=[human_intervention],
)

## Human interventions needed: 1 (minimum)

#response = agent.run("""Please generate me a professional email signature""")
#print(response.content)

## Human interventions needed: 0

response = agent.run("""# Please generate me a professional email signature for:
full name: Sarah Johnson
job title: Marketing Manager
company name: TechCorp Solutions
phone number: (555) 123-4567
email address: sarah.johnson@techcorp.com
website: www.techcorp.com
LinkedIn profile: linkedin.com/in/sarahjohnson""")
print(response.content)

human_interventions = sum(1 for call in response.tools if call.tool_name == "human_intervention")
print(f"Human interventions: {human_interventions}")



#
