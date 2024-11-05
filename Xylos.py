import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from dotenv import load_dotenv
from langchain_community.llms import GPT4All
import os
import logging
from typing import Dict, Any, Callable
from langchain_core.output_parsers import StrOutputParser
from logging.handlers import RotatingFileHandler
from log_manager import LogManager

# Add these imports at the top of your main file
from config_manager import ConfigManager
from admin_interface import AdminInterface

# Initialize ConfigManager and AdminInterface
config_manager = ConfigManager()
admin_interface = AdminInterface(config_manager)
log_manager = LogManager()
logger = log_manager.logger

def create_chain(question_type: str, chatgpt, claude, local):
    """Create appropriate chain based on question type."""
    categories = config_manager.get_categories()
    config = categories[question_type]

    prompt_template = config["prompt_template"]
    model_name = config["model"]

    if model_name == "claude":
        prompt = PromptTemplate.from_template(prompt_template)
        return prompt | claude
    elif model_name == "chatgpt":
        prompt = PromptTemplate.from_template(prompt_template)
        return prompt | chatgpt
    else:  # local model
        prompt = PromptTemplate.from_template(prompt_template)
        def local_chain(input_dict):
            question = input_dict["input"]
            formatted_prompt = prompt.format(input=question)
            return local(formatted_prompt)
        return local_chain

# Load environment variables and set up logging
load_dotenv()
logging.basicConfig(level=logging.INFO)
# Configure logging

# Initialize session state for models instead of chain
if 'models' not in st.session_state:
    st.session_state.models = None

def validate_api_keys():
    missing_keys = []
    if not os.getenv("OPENAI_API_KEY"):
        missing_keys.append("OPENAI_API_KEY")
    if not os.getenv("ANTHROPIC_API_KEY"):
        missing_keys.append("ANTHROPIC_API_KEY")
    
    if missing_keys:
        raise EnvironmentError(
            f"Missing required API keys: {', '.join(missing_keys)}. "
            "Please ensure these are set in your .env file"
        )

def initialize_models():
    try:
        validate_api_keys()
        
        chatgpt = ChatOpenAI(
            temperature=0.7,
            model="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        claude = ChatAnthropic(
            temperature=0.7,
            model="claude-3-haiku-20240307",
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        
        local = GPT4All(
            model=r"C:\Users\andre\AppData\Local\nomic.ai\GPT4All\Meta-Llama-3-8B-Instruct.Q4_0.gguf",
            allow_download=False,
            device='cpu'  # Force CPU usage instead of CUDA
        )
        
        logger.info("Models initialized successfully")
        return chatgpt, claude, local
    
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        st.error(f"Error initializing models: {str(e)}")
        return None, None, None  # Added 'None' for 'local' to match return values

def analyze_question_type(question: str, config_manager: ConfigManager) -> str:
    """
    Analyze the question to determine its type.
    
    Args:
        question: The user's question
        config_manager: Instance of ConfigManager containing category configurations
        
    Returns:
        str: The determined category type
    """
    question_lower = question.lower()
    categories = config_manager.get_categories()
    
    # Calculate matches for each category
    category_matches = {
        category: sum(1 for keyword in config["keywords"] 
                      if keyword.lower() in question_lower)
        for category, config in categories.items()
    }
    
    # Find category with most keyword matches
    max_matches = max(category_matches.values())
    matching_categories = [
        category for category, matches in category_matches.items()
        if matches == max_matches
    ]

    # If no clear match is found, return "general"
    if max_matches == 0 or len(matching_categories) > 1:
        return "general"
        
    return matching_categories[0]

def process_question(question: str) -> str:
    """
    Process a single question and return the response.
    
    Args:
        question: The user's question
        
    Returns:
        str: The AI-generated response
    """
    try:
        # Initialize models if not already done
        if not st.session_state.models:
            chatgpt, claude, local = initialize_models()
            if not chatgpt or not claude or not local:
                logger.error("Failed to initialize models")
                return "Error: Failed to initialize models"
            st.session_state.models = (chatgpt, claude, local)
        else:
            chatgpt, claude, local = st.session_state.models
        
        # Get config manager instance
        if not hasattr(st.session_state, 'config_manager'):
            st.session_state.config_manager = ConfigManager()
        
        # Analyze question type using config manager
        question_type = analyze_question_type(question, st.session_state.config_manager)
        logger.info(f"Question type determined: {question_type}")
        
        # Create appropriate chain for this specific question
        chain = create_chain(question_type, chatgpt, claude, local)
        
        try:
            response = chain.invoke({"input": question})
            logger.info("Successfully generated response")
            
            # Handle different response types
            if isinstance(response, str):
                return response
            elif hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
                
        except AttributeError as ae:
            error_msg = f"Chain invocation error: {str(ae)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
    except Exception as e:
        error_msg = f"Error processing question: {str(e)}"
        logger.error(error_msg)
        return error_msg

# Streamlit UI
st.title("Interactive Question Answering System")
st.write("Ask questions about finance, literature, or any other topic!")

# Add this to your Streamlit UI section
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Question Answering", "Admin"])

if page == "Question Answering":
    # Your existing question answering interface code here
    pass

# Replace the admin interface section with:
elif page == "Admin":
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False

    if not st.session_state.admin_authenticated:
        admin_interface.render_login()  # Changed from render_admin_login to render_login
    else:
        admin_interface.render_admin_interface()

# Add example questions to help users
st.sidebar.header("Example Questions")
st.sidebar.write("""
- What are the best strategies for long-term investing?
- Can you analyze the themes in Pride and Prejudice?
- How do I create a diversified portfolio?
- What are the major literary movements of the 20th century?
""")

# Main input area
question = st.text_input("Enter your question:", key="question_input")

if st.button("Get Analysis"):
    if question:
        with st.spinner("Analyzing your question..."):
            try:
                response = process_question(question)
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a question before submitting.")

# Add a reset button to clear the session state
if st.button("Reset Session"):
    st.session_state.models = None
    st.success("Session reset successfully!")

# Footer with information
st.markdown("---")
st.markdown("""
*This system uses advanced AI to provide detailed analyses across various topics. 
Responses are generated using a combination of GPT-3.5 and Claude models, with intelligent routing based on question content.*
""")
