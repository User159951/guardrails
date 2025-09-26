import os
import asyncio
from textwrap import dedent
from datetime import datetime
from agno.agent import Agent
from agno.models.xai import xAI
from agno.tools.reasoning import ReasoningTools
from agno.tools.googlesearch import GoogleSearchTools
from presidio_analyzer import AnalyzerEngine
import os
import dotenv

dotenv.load_dotenv()

XAI_API_KEY = os.getenv("XAI_API_KEY")
# Personally identifiable information (PII) Detection

# ---------- TOOLS ----------

def get_date_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Instantiate Presidio's AnalyzerEngine once for efficiency
analyzer = AnalyzerEngine()

def detect_pii(text: str):
    """
    Detect PII entities in the given text using Presidio.
    Returns a list of detected entities with their type, position, and value.
    """
    results = analyzer.analyze(text=text, entities=[], language='en')
    return [
        {
            "start": r.start,
            "end": r.end,
            "score": r.score,
            "text": text[r.start:r.end]
        }
        for r in results
    ]

# ---------- AGNO AGENT ----------

agent = Agent(
    name="Healthcare Tech Assistant",
    model=xAI(id='grok-3-mini', api_key=XAI_API_KEY),
    instructions=dedent("""
        You are an assistant specialized in new healthcare technologies in Europe.
        - If the user explicitly asks for the current date or time, call 'get_date_time'.
        - If the user asks for up-to-date info (e.g., news, research), call 'google_search'.
        - If the user asks to detect PII (Personally identifiable information) in text, call 'detect_pii'.
        - Always include the tools result in your final answer.
        """),
    tools=[GoogleSearchTools(), get_date_time, detect_pii],
)

# ---------- EXAMPLE USAGE ----------

if __name__ == "__main__":
    response = agent.run("Detect PII in: Jane Smith, phone 555-123-4567, email jane@health.org")
    print(response.content)
