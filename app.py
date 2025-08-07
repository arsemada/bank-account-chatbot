import os
import streamlit as st
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from enum import Enum, auto

# --- Configuration and Initialization ---

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Error: GOOGLE_API_KEY not found in environment variables.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# --- Initialize Session State ---

if "conversation_state" not in st.session_state:
    st.session_state.conversation_state = "IDLE"
if "account_details" not in st.session_state:
    st.session_state.account_details = {}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "exact_match_cache" not in st.session_state:
    st.session_state.exact_match_cache = {}

# --- ChromaDB Setup ---
@st.cache_resource
def setup_chromadb():
    embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=GOOGLE_API_KEY, model_name="models/embedding-001")
    database_path = "./chromadb_data"
    try:
        client = chromadb.Client(Settings(persist_directory=database_path, is_persistent=True))
        collection = client.get_or_create_collection(name="chatbot_responses", embedding_function=embedding_function)
        return collection, embedding_function
    except Exception as e:
        st.error(f"An error occurred while setting up persistent ChromaDB: {e}")
        st.stop()

collection, embedding_function = setup_chromadb()

# --- Chatbot Functions ---

def get_semantic_cached_response(query_text, similarity_threshold=0.9):
    results = collection.query(query_texts=[query_text], n_results=1)
    if not results or not results['ids'] or not results['ids'][0]:
        return None
    distance = results['distances'][0][0]
    cached_document = results['documents'][0][0]
    if distance <= (1 - similarity_threshold):
        return cached_document
    return None

def store_response(query_text, bot_response):
    st.session_state.exact_match_cache[query_text] = bot_response
    try:
        current_count = collection.count()
        unique_id = f"response_{current_count + 1}"
        collection.add(
            documents=[bot_response],
            metadatas=[{"query": query_text}],
            ids=[unique_id]
        )
    except Exception as e:
        st.error(f"Error storing response in ChromaDB: {e}")

# --- State Management for Account Opening ---
class ChatState(Enum):
    IDLE = auto()
    ASK_NAME = auto()
    ASK_EMAIL = auto()
    ASK_ACCOUNT_TYPE = auto()
    CONFIRMATION = auto()
    COMPLETED = auto()

def handle_account_opening_flow(user_input):
    state = st.session_state.conversation_state
    account_details = st.session_state.account_details
    response_text = ""
    
    if state == "ASK_NAME":
        account_details['name'] = user_input
        st.session_state.conversation_state = "ASK_EMAIL"
        response_text = f"Thank you, {account_details['name']}. What is your email address?"
    elif state == "ASK_EMAIL":
        account_details['email'] = user_input
        st.session_state.conversation_state = "ASK_ACCOUNT_TYPE"
        response_text = "What type of account would you like to open? (e.g., Checking, Savings)"
    elif state == "ASK_ACCOUNT_TYPE":
        account_details['account_type'] = user_input
        st.session_state.conversation_state = "CONFIRMATION"
        response_text = (
            f"Please confirm your details:\n"
            f"Name: {account_details['name']}\n"
            f"Email: {account_details['email']}\n"
            f"Account Type: {account_details['account_type']}\n"
            "Is this correct? (yes/no)"
        )
    elif state == "CONFIRMATION":
        if user_input.lower() in ["yes", "y", "correct"]:
            response_text = "Thank you! Your account opening request has been submitted. A representative will be in touch shortly."
            st.session_state.conversation_state = "COMPLETED"
        else:
            response_text = "No problem. Let's start over. What is your full name?"
            st.session_state.conversation_state = "ASK_NAME"
    elif state == "COMPLETED":
        response_text = "Your request has been submitted. Feel free to ask any other questions!"
        st.session_state.conversation_state = "IDLE"
        
    st.session_state.account_details = account_details
    return response_text

def new_chat():
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am a Bank Account Chatbot. How may I help you?"}]
    st.session_state.conversation_state = "IDLE"
    st.session_state.account_details = {}

# --- UI Layout and Logic ---

# Custom CSS for a modern, beautiful UI
st.markdown("""
<style>
    .reportview-container {
        background: #0E1117;
    }
    .st-emotion-cache-1g6x8u { /* Chat bubble container */
        background-color: #262730;
        border-radius: 20px;
        padding: 10px;
    }
    .st-emotion-cache-1r4qj8m { /* User chat bubble */
        background-color: #007bff;
        color: white;
    }
    .st-emotion-cache-1k80p4i { /* Assistant chat bubble */
        background-color: #3b3c3e;
        color: white;
    }
    .st-emotion-cache-1v0t34p { /* Message content */
        color: white;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .st-emotion-cache-j7qwjs { /* Chat input container */
        background-color: #262730;
    }
    .st-emotion-cache-lgl296 { /* Chat input text */
        background-color: #1e1e1e;
        color: white;
    }
    .st-emotion-cache-9u331j { /* Chat input button */
        color: #007bff;
    }
</style>
""", unsafe_allow_html=True)

# Main title and welcome message logic
st.title("🏦 My AI Bank Chatbot")

# Sidebar for new chat and FAQ
with st.sidebar:
    st.header("My Bank")
    st.markdown("---")
    st.subheader("Your Personal Banking Assistant")
    if st.button("✨ New Chat"):
        new_chat()
        st.rerun()

    st.markdown("---")
    st.subheader("Quick Answers (FAQ)")
    with st.expander("What are your operating hours?"):
        st.markdown("Our online support is available 24/7. Our branches are open from 8 AM to 5 PM, Monday to Friday.")
    with st.expander("How do I report a lost card?"):
        st.markdown("Please call our support line or visit your nearest branch immediately to report a lost or stolen card.")
    with st.expander("What do I need to open an account?"):
        st.markdown("To open an account, you'll generally need a valid government-issued ID (such as a driver's license or passport), proof of address. You can start the process by typing 'I want to open an account'.")

# Initialize chat messages if empty
if not st.session_state.messages:
    new_chat()

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input and chatbot response logic
if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # State Handling and Intent Recognition
        if st.session_state.conversation_state != "IDLE":
            response = handle_account_opening_flow(prompt)
        elif "open an account" in prompt.lower() or "open a bank account" in prompt.lower():
            st.session_state.conversation_state = "ASK_NAME"
            st.session_state.account_details = {}
            response = "Great! To get started, what is your full name?"
        # Fallback to Caching and Gemini API
        else:
            if prompt in st.session_state.exact_match_cache:
                response = st.session_state.exact_match_cache[prompt]
            else:
                semantic_response = get_semantic_cached_response(prompt)
                if semantic_response:
                    response = semantic_response
                    st.session_state.exact_match_cache[prompt] = response
                else:
                    try:
                        response_from_gemini = model.generate_content(prompt)
                        response = response_from_gemini.text
                        store_response(prompt, response)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        response = "I'm sorry, I'm having trouble generating a response right now. Please try again later."
        
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})