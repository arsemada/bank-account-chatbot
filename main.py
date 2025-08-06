import os
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from enum import Enum, auto # Import Enum for state management

# --- Configuration and Initialization ---

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not found in environment variables.")
    exit()

genai.configure(api_key=GOOGLE_API_KEY)
print("Gemini API configured successfully.")

embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=GOOGLE_API_KEY, model_name="models/embedding-001")
database_path = "./chromadb_data"

try:
    client = chromadb.Client(Settings(persist_directory=database_path, is_persistent=True))
    collection = client.get_or_create_collection(name="chatbot_responses", embedding_function=embedding_function)
    print(f"ChromaDB collection 'chatbot_responses' created or retrieved successfully at {database_path}.")
except Exception as e:
    print(f"An error occurred while setting up persistent ChromaDB: {e}")
    exit()

EXACT_MATCH_CACHE = {}

# --- Functions for Caching (unchanged) ---

def get_semantic_cached_response(query_text, similarity_threshold=0.9):
    results = collection.query(
        query_texts=[query_text],
        n_results=1
    )

    if not results or not results['ids'] or not results['ids'][0]:
        return None

    distance = results['distances'][0][0]
    cached_document = results['documents'][0][0]

    if distance <= (1 - similarity_threshold):
        print(f"DEBUG: Semantic cache hit! Distance: {distance:.4f} (Threshold: {(1 - similarity_threshold):.4f})")
        return cached_document

    return None

def store_response(query_text, bot_response):
    EXACT_MATCH_CACHE[query_text] = bot_response
    print("DEBUG: Response stored in in-memory exact match cache.")

    try:
        current_count = collection.count()
        unique_id = f"response_{current_count + 1}" 

        collection.add(
            documents=[bot_response],
            metadatas=[{"query": query_text}],
            ids=[unique_id]
        )
        print(f"DEBUG: New response stored in ChromaDB with ID: {unique_id}.")
    except Exception as e:
        print(f"Error storing response in ChromaDB: {e}")

# --- New: State Management for Account Opening ---

class ChatState(Enum):
    IDLE = auto()
    ASK_NAME = auto()
    ASK_EMAIL = auto()
    ASK_ACCOUNT_TYPE = auto()
    CONFIRMATION = auto()
    COMPLETED = auto()

conversation_state = ChatState.IDLE
account_details = {}

# --- New: Function to handle the stateful conversation ---

def handle_account_opening_flow(user_input):
    global conversation_state, account_details
    response_text = ""

    if conversation_state == ChatState.ASK_NAME:
        account_details['name'] = user_input
        conversation_state = ChatState.ASK_EMAIL
        response_text = f"Thank you, {account_details['name']}. What is your email address?"

    elif conversation_state == ChatState.ASK_EMAIL:
        account_details['email'] = user_input
        conversation_state = ChatState.ASK_ACCOUNT_TYPE
        response_text = "What type of account would you like to open? (e.g., Checking, Savings)"

    elif conversation_state == ChatState.ASK_ACCOUNT_TYPE:
        account_details['account_type'] = user_input
        conversation_state = ChatState.CONFIRMATION
        response_text = (
            f"Please confirm your details:\n"
            f"Name: {account_details['name']}\n"
            f"Email: {account_details['email']}\n"
            f"Account Type: {account_details['account_type']}\n"
            "Is this correct? (yes/no)"
        )

    elif conversation_state == ChatState.CONFIRMATION:
        if user_input.lower() in ["yes", "y", "correct"]:
            # Here you would typically process the request (e.g., save to a database)
            response_text = "Thank you! Your account opening request has been submitted. A representative will be in touch shortly."
            conversation_state = ChatState.COMPLETED
        else:
            response_text = "No problem. Let's start over. What is your full name?"
            conversation_state = ChatState.ASK_NAME

    elif conversation_state == ChatState.COMPLETED:
        response_text = "Your request has been submitted. Feel free to ask any other questions!"
        conversation_state = ChatState.IDLE # Reset the state

    return response_text

# --- Chatbot Logic (modified to handle state) ---

def chatbot():
    global conversation_state, account_details
    print("\nWelcome to the Bank Account Opening Chatbot!")
    print("You can start by asking a general question or by saying 'I want to open an account'.")
    
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        # --- State Handling Logic ---
        if conversation_state != ChatState.IDLE:
            response = handle_account_opening_flow(user_input)
            print(f"Bot: {response}")
            # If the flow is complete, we should now allow the loop to process the next input normally.
            if conversation_state == ChatState.IDLE:
                continue
            continue
            
        # --- Intent Recognition for general queries ---
        if "open an account" in user_input.lower() or "open a bank account" in user_input.lower():
            conversation_state = ChatState.ASK_NAME
            print("Bot: Great! To get started, what is your full name?")
            continue
            
        # --- Fallback to Caching and Gemini API for general questions ---
        if user_input in EXACT_MATCH_CACHE:
            print(f"Bot (from exact cache): {EXACT_MATCH_CACHE[user_input]}")
            continue
        
        semantic_cached_response = get_semantic_cached_response(user_input)
        
        if semantic_cached_response:
            print(f"Bot (from semantic cache): {semantic_cached_response}")
            EXACT_MATCH_CACHE[user_input] = semantic_cached_response
        else:
            try:
                print("DEBUG: Calling Gemini API for a new response...")
                response = model.generate_content(user_input)
                new_response = response.text
                print(f"Bot (from Gemini): {new_response}")
                
                store_response(user_input, new_response)
                
            except Exception as e:
                print(f"An error occurred while calling the Gemini API: {e}")
                print("Bot: I'm sorry, I'm having trouble generating a response right now. Please try again later.")

if __name__ == "__main__":
    chatbot()