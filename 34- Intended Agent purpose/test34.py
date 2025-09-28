from dotenv import load_dotenv
import os
from agno.agent import Agent
from datetime import datetime
from textwrap import dedent
from agno.models.xai import xAI
from agno.vectordb.pgvector import PgVector
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.knowledge.markdown import MarkdownKnowledgeBase
from agno.embedder.mistral import MistralEmbedder

load_dotenv()

XAI_API_KEY = os.getenv("XAI_API_KEY")

# Model and System cards, including clear documentation of intended agent purpose and limitations

# ---------- Initializing Knowledge Base ----------

knowledge_base = MarkdownKnowledgeBase(
    path="Files/system_card.md",
    vector_db=PgVector(
        table_name="system_card",
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
        embedder=MistralEmbedder(api_key=os.getenv('MISTRAL_API_KEY')),
    )
)

knowledge_base.load(recreate=True)


# -------------------- TOOLS --------------------

def get_date_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# -------------------- MAIN AGENT --------------------

main_agent = Agent(
    name="Helpful Assistant",
    model=xAI(id='grok-3-mini', api_key=XAI_API_KEY),
    instructions=dedent("""
        You are a Healthcare Tech Assistant.
        - Use the knowledge base as your main reference.
        - If the user asks about anything outside knowledge base boundaries, you must strictly decline
        - Do not answer questions outside this domain, even if tools could provide an answer.
        - Always be clear and concise.
    """),
    tools=[DuckDuckGoTools(), get_date_time],
    knowledge=knowledge_base,
)


#main_agent.print_response('What are latest AI News in healthcare in europe')
main_agent.print_response('what are the best travelling places in Italy')
