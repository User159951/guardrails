import os
from agno.agent import Agent
from agno.tools.e2b import E2BTools
from agno.models.xai import xAI
import dotenv

dotenv.load_dotenv()
XAI_API_KEY = os.getenv("XAI_API_KEY")

# Implement separate execution environments
e2b_tools = E2BTools(
    timeout=600, # 10 minutes timeout (in seconds)
)

agent = Agent(
    name="Code Execution Sandbox",
    id="e2b-sandbox",
    model=xAI(id='grok-3-mini', api_key=XAI_API_KEY),
    tools=[e2b_tools],
    markdown=True,
    instructions=[
        "You are an expert at writing and validating Python code using a secure E2B sandbox environment.",
        "Your primary purpose is to:",
        "1. Write clear, efficient Python code based on user requests",
        "2. Execute and verify the code in the E2B sandbox",
        "3. Share the complete code with the user, as this is the main use case",
        "4. Provide thorough explanations of how the code works",
        "",
        "You can use these tools:",
        "1. Run Python code (run_python_code)",
        "2. Upload files to the sandbox (upload_file)",
        "3. Download files from the sandbox (download_file_from_sandbox)",
        "4. Generate and add visualizations as image artifacts (download_png_result)",
        "5. List files in the sandbox (list_files)",
        "6. Read and write file content (read_file_content, write_file_content)",
        "7. Start web servers and get public URLs (run_server, get_public_url)",
        "8. Manage the sandbox lifecycle (set_sandbox_timeout, get_sandbox_status, shutdown_sandbox)",
    ],
)

# Example: Generate Fibonacci numbers
agent.print_response(
    "Write Python code to generate the first 10 Fibonacci numbers and calculate their sum and average"
)
