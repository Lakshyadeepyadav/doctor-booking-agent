import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY environment variable is not set")

model = ChatGroq(
    api_key=api_key,
    model="llama-3.1-8b-instant",
    #model="gpt-oss-20b",
    temperature=0.0,
    max_retries=2,
)
