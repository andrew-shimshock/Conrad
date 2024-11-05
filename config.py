"""
Configuration settings for the LLM Router application.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Model configurations
OPENAI_MODEL = "gpt-3.5-turbo"
ANTHROPIC_MODEL = "claude-2"

# Classification keywords (can be modified as needed)
OPENAI_KEYWORDS = ['science', 'math', 'coding', 'technical']
ANTHROPIC_KEYWORDS = ['ethics', 'philosophy', 'creative', 'writing']

# Error messages
ERROR_MISSING_API_KEYS = "Missing API keys. Please check your .env file."
ERROR_CLASSIFICATION_FAILED = "Classification failed. Defaulting to OpenAI."
