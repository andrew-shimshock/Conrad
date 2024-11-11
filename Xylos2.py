import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import ConversationChain
from dotenv import load_dotenv
from langchain_community.llms import GPT4All
import os
import logging
from typing import Dict, Any, Callable, Optional
from logging.handlers import RotatingFileHandler
from log_manager import LogManager
from config_manager import ConfigManager
from admin_interface import AdminInterface
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory
# Load environment variables at startup
load_dotenv()

def initialize_session_var(var_name: str, initializer: Any) -> Any:
    """
    Initialize a single session state variable with error handling.
    
    Args:
        var_name: Name of the session state variable
        initializer: Class or function to initialize the variable
        
    Returns:
        Initialized variable value
    """
    try:
        if isinstance(initializer, type):
            # If initializer is a class, instantiate it
            return initializer()
        elif callable(initializer):
            # If initializer is a function, call it
            return initializer()
        else:
            # If initializer is a value, return it
            return initializer
    except Exception as e:
        logging.error(f"Error initializing {var_name}: {str(e)}")
        return None

def init_session_state():
    """
    Initialize all session state variables with proper error handling and logging.
    """
    # Define initialization map with proper typing
    session_vars: Dict[str, Any] = {
        'config_manager': lambda: ConfigManager(),
        'log_manager': lambda: LogManager(),
        'models': None,
        'message_history': lambda: StreamlitChatMessageHistory(key="chat_messages"),
        'current_page': "Question Answering",
        'admin_authenticated': False,
        'conversation_id': 0,
        'memories': {},
        'chat_history': []
    }
    
    # Initialize each session variable
    for var_name, initializer in session_vars.items():
        if var_name not in st.session_state:
            st.session_state[var_name] = initialize_session_var(var_name, initializer)
    
    # Initialize admin interface after config manager is ready
    if 'admin_interface' not in st.session_state and st.session_state.config_manager:
        st.session_state.admin_interface = initialize_session_var(
            'admin_interface',
            lambda: AdminInterface(st.session_state.config_manager)
        )

def initialize_memories():
    """
    Initialize conversation memories for each model.
    """
    if 'memories' not in st.session_state or not st.session_state.memories:
        st.session_state.memories = {
            model_name: ConversationBufferMemory(
                memory_key="history",
                return_messages=True,
                chat_memory=st.session_state.message_history
            ) for model_name in ['chatgpt', 'claude', 'local']
        }

def initialize_models() -> tuple[Optional[ChatOpenAI], Optional[ChatAnthropic], Optional[GPT4All]]:
    """
    Initialize language models with proper error handling.
    
    Returns:
        Tuple of initialized models (ChatGPT, Claude, Local) or None values if initialization fails
    """
    try:
        # Validate API keys
        required_keys = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY")
        }
        
        missing_keys = [key for key, value in required_keys.items() if not value]
        if missing_keys:
            raise EnvironmentError(f"Missing required API keys: {', '.join(missing_keys)}")
        
        # Initialize memories if needed
        initialize_memories()
        
        # Initialize models
        chatgpt = ChatOpenAI(
            temperature=0.7,
            model="gpt-3.5-turbo",
            api_key=required_keys["OPENAI_API_KEY"]
        )
        
        claude = ChatAnthropic(
            temperature=0.7,
            model="claude-3-haiku-20240307",
            anthropic_api_key=required_keys["ANTHROPIC_API_KEY"]
        )
        
        local = GPT4All(
            model=r"Meta-Llama-3-8B-Instruct.Q4_0.gguf",
            allow_download=False,
            device='cpu'
        )
        
        logging.info("Models initialized successfully")
        return chatgpt, claude, local
        
    except Exception as e:
        logging.error(f"Error initializing models: {str(e)}")
        return None, None, None

def display_chat_history():
    """
    Display the chat history in the Streamlit UI with proper formatting.
    """
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            with st.container():
                if message['role'] == 'user':
                    st.write(f"🧑 You: {message['content']}")
                else:
                    st.write(f"🤖 Assistant: {message['content']}")

def process_question(question: str) -> Optional[str]:
    """
    Process a question and maintain conversation history.
    
    Args:
        question: The user's question
        
    Returns:
        Optional[str]: The model's response or None if processing fails
    """
    try:
        if not st.session_state.models:
            models = initialize_models()
            if not all(models):
                raise RuntimeError("Failed to initialize one or more models")
            st.session_state.models = models
        
        chatgpt, claude, local = st.session_state.models
        
        # Determine question type and select appropriate model
        question_type = analyze_question_type(question)
        logging.info(f"Question type determined: {question_type}")
        
        # Create and execute chain
        chain = create_chain(question_type, chatgpt, claude, local)
        response = chain.predict(input=question)
        
        # Update chat history
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        return response
        
    except Exception as e:
        logging.error(f"Error processing question: {str(e)}")
        return f"Error: {str(e)}"
def create_chain(question_type: str, chatgpt, claude, local):
    """Create appropriate chain based on question type with fallback options."""
    try:
        categories = st.session_state.config_manager.get_categories()
        config = categories[question_type]
        model_name = config["model"]

        chat_prompt = ChatPromptTemplate.from_template("""
        Previous conversation:
        {history}
        
        Human: {input}
        Assistant: Let me help you with that.
        """)

        # Create memory for the chain
        memory = st.session_state.memories.get(model_name, 
                                             st.session_state.memories["chatgpt"])

        # Model selection with fallback logic
        if model_name == "claude" and claude:
            model = claude
        elif model_name == "chatgpt" and chatgpt:
            model = chatgpt
        elif local:
            model = local
        else:
            # Fallback to any available model
            model = next(m for m in [chatgpt, claude, local] if m is not None)
            if not model:
                raise RuntimeError("No available models")

        return ConversationChain(
            llm=model,
            prompt=chat_prompt,
            memory=memory,
            verbose=True
        )

    except Exception as e:
        logging.error(f"Error creating chain: {str(e)}")
        raise RuntimeError(f"Failed to create conversation chain: {str(e)}")
def main():
    """
    Main application function with streamlined UI and error handling.
    Implements a sophisticated chat interface with proper state management
    and real-time response handling.
    """
    # Initialize session state
    init_session_state()
    
    # Page configuration
    st.set_page_config(
        page_title="AI Question Answering System",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Main UI
    st.title("Interactive Question Answering System")
    
    # Sidebar Navigation and Controls
    with st.sidebar:
        st.title("Navigation & Controls")
        page = st.radio(
            "Select Interface",
            ["Question Answering", "Admin"],
            key="page_selector"
        )
        
        # Model Status Indicator
        st.subheader("System Status")
        if st.session_state.models:
            st.success("🟢 Models Loaded")
        else:
            st.warning("🟡 Models Not Initialized")
        
        # Example Questions Section
        st.subheader("Example Questions")
        example_questions = {
            "Finance": "What are the key factors to consider for retirement planning?",
            "Literature": "Analyze the symbolism in Shakespeare's Macbeth.",
            "Technology": "Explain the concept of distributed computing.",
            "Science": "How does quantum entanglement work?"
        }
        
        for category, question in example_questions.items():
            if st.button(f"Try {category}", key=f"example_{category.lower()}"):
                st.session_state.current_question = question
        
        # Reset Session Button
        if st.button("Reset Session", key="reset_session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            init_session_state()
            st.success("Session reset successfully!")
            st.experimental_rerun()
    
    # Main Content Area
    if page == "Question Answering":
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Chat Interface")
            
            # Chat History Display
            with st.container():
                st.markdown("### Conversation History")
                display_chat_history()
            
            # Question Input Area
            st.markdown("### Ask Your Question")
            question = st.text_area(
                "Enter your question:",
                value=st.session_state.get('current_question', ''),
                key=f"question_input_{st.session_state.conversation_id}",
                height=100
            )
            
            # Control Buttons
            col_submit, col_clear = st.columns([1, 1])
            with col_submit:
                submit_button = st.button(
                    "Submit Question",
                    key=f"submit_{st.session_state.conversation_id}"
                )
            with col_clear:
                clear_button = st.button(
                    "Clear Input",
                    key=f"clear_{st.session_state.conversation_id}"
                )
            
            if clear_button:
                st.session_state.current_question = ""
                st.experimental_rerun()
            
            if submit_button and question:
                with st.spinner("Processing your question..."):
                    try:
                        response = process_question(question)
                        if response:
                            st.session_state.conversation_id += 1
                            st.session_state.current_question = ""
                            st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")
                        logging.error(f"Question processing error: {str(e)}")
            
        with col2:
            # Information and Stats Panel
            st.subheader("System Information")
            with st.expander("About the System", expanded=True):
                st.markdown("""
                This advanced AI system combines multiple language models:
                - **GPT-3.5**: General knowledge and analysis
                - **Claude**: Complex reasoning and ethical considerations
                - **Local Model**: Offline processing capabilities
                
                The system automatically routes your questions to the most
                appropriate model based on content analysis.
                """)
            
            # Conversation Statistics
            if st.session_state.chat_history:
                st.subheader("Conversation Stats")
                total_messages = len(st.session_state.chat_history)
                user_messages = sum(1 for msg in st.session_state.chat_history 
                                  if msg['role'] == 'user')
                
                st.metric("Total Interactions", total_messages // 2)
                st.metric("Questions Asked", user_messages)
    
    elif page == "Admin":
        if not st.session_state.admin_authenticated:
            with st.container():
                st.subheader("Admin Login")
                st.session_state.admin_interface.render_login()
        else:
            st.session_state.admin_interface.render_admin_interface()
    
    # Footer
    st.markdown("---")
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style='text-align: center'>
            <p><em>Powered by advanced AI models with intelligent routing</em></p>
            <p>© 2024 AI Question Answering System</p>
            </div>
            """, unsafe_allow_html=True)

def analyze_question_type(question: str) -> str:
    """
    Analyze the question content to determine its category and appropriate model.
    
    Args:
        question: The user's question text
        
    Returns:
        str: Determined category for routing
    """
    question_lower = question.lower()
    categories = st.session_state.config_manager.get_categories()
    
    # Calculate keyword matches for each category
    category_matches = {
        category: sum(1 for keyword in config["keywords"] 
                     if keyword.lower() in question_lower)
        for category, config in categories.items()
    }
    
    # Find best matching category
    max_matches = max(category_matches.values())
    matching_categories = [
        category for category, matches in category_matches.items()
        if matches == max_matches
    ]
    
    # Return most appropriate category or default to general
    return matching_categories[0] if len(matching_categories) == 1 else "general"

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        logging.error(f"Application error: {str(e)}")