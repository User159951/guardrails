import os
from pathlib import Path
from pyairtable import Api
from agno.team import Team
from textwrap import dedent
from agno.agent import Agent
from datetime import datetime
from dotenv import load_dotenv
from agno.models.xai import xAI
from agno.memory.v2.memory import Memory
from agno.models.openai import OpenAIChat
from agno.models.mistral import MistralChat
from agno.vectordb.pgvector import PgVector
from agno.tools.csv_toolkit import CsvTools
from agno.tools.reasoning import ReasoningTools
from agno.embedder.mistral import MistralEmbedder
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.knowledge.markdown import MarkdownKnowledgeBase

load_dotenv()

# Isolate authentication information required for a tool from the model
AIRTABLE_API_KEY = os.getenv('AIRTABLE_API_KEY')
AIRTABLE_BASE_ID = os.getenv('AIRTABLE_BASE_ID')
AIRTABLE_EMPLOYEES_TABLE_ID = os.getenv('AIRTABLE_EMPLOYEES_TABLE_ID')

def get_all_employees() -> list[dict]:
    """Get all records from the table
    Returns:
        - A list of dictionaries of records
    """
    api = Api(AIRTABLE_API_KEY)
    table = api.table(AIRTABLE_BASE_ID, AIRTABLE_EMPLOYEES_TABLE_ID)
    return table.all()

def create_onboarding_module():
    # Initializing Knowledge Base
    knowledge_base = MarkdownKnowledgeBase(
        path="CompanyKnowledge/onboarding_data.md",
        vector_db=PgVector(
            table_name="onboarding_data",
            db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
            embedder=MistralEmbedder(api_key=os.getenv('MISTRAL_API_KEY'))
        )
    )
    knowledge_base.load(recreate=True)

    # Initializing Memory
    memory = Memory(
        model=xAI(id="grok-3-mini"),
        #model=OpenAIChat(id='gpt-4o-mini')
        db=SqliteMemoryDb(table_name="user_memories", db_file='Database/onboarding_memory.db')
    )

    OnboardingAssistant = Agent(
        name='Onboarding Assistant',
        model=MistralChat(id='mistral-large-latest', api_key=os.getenv('MISTRAL_API_KEY')),
        description="You are an Onboarding Assistant designed to assist HR in onboarding new employees.",
        instructions=dedent("""
        ## Goal
        Your primary goal is to assist HR managers in onboarding new hires efficiently and professionally. You will:
        - Generate personalized onboarding plans for employees.
        - Provide professional onboarding paths including all required tasks, milestones, and timelines.
        - Reference the company's onboarding knowledge base (which contains policies, procedures, and role-specific information for each department).
        - Track progress and prepare resources HR can share with the employee.
        - Answer HR's questions about onboarding, company policies, and role-specific requirements.

        ## Available Tools
        - `get_all_employees`: Returns a list of employees.
        Use this if HR asks about onboarding for a specific employee or requests a plan for an existing employee.

        ## How to Work
        1. Always base answers and plans on the official company onboarding knowledge base.
        2. When HR asks for an onboarding plan for a specific employee:
           - Retrieve their role, department, and start date (either from the request or by using `get_all_employees` if not provided).
           - Tailor the onboarding plan to their position and department using the knowledge base.
           - Include:
             - Welcome and introduction steps.
             - Required document completion.
             - Department orientation.
             - Role-specific training.
             - Compliance and security briefings.
             - Milestones for first week, first month, and first 90 days.

        3. When answering HR's general questions:
           - Always give accurate, concise, and professional responses using the knowledge base.
        4. If information is missing:
           - Ask the HR user for clarification before proceeding.

        ## Tone & Style
        - Professional, clear, and actionable.
        - Structured so HR can directly share the plan with a new hire if needed.
        - Avoid informal or ambiguous language.

        ## Example Use Cases
        - "Give me an onboarding plan for Jane Doe starting next Monday in Engineering as a Software Engineer."
        - "What's the dress code for Marketing?"
        - "Show me the onboarding steps for all new Account Executives."
        """),
        tools=[ReasoningTools(add_instructions=True), get_all_employees],
        #debug_mode=True,
        memory=memory,
        enable_agentic_memory=True,
        add_history_to_messages=True,
        knowledge=knowledge_base,
    )
