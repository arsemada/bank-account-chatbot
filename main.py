import os
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
from chromadb.config import Settings # Import Settings for persistent client

# --- Configuration and Initialization ---

# Load environment variables from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not found in environment variables.")
    exit()

# Configure the Gemini API
genai.configure(api_key=GOOGLE_API_KEY)
print("Gemini API configured successfully.")

# Define the embedding function for ChromaDB to use the Gemini embedding model
embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=GOOGLE_API_KEY, model_name="models/embedding-001")

# Define the path for the persistent ChromaDB database
database_path = "./chromadb_data" # This will create a 'chromadb_data' folder in your project directory

# Initialize the ChromaDB client with persistence
try:
    client = chromadb.Client(Settings(persist_directory=database_path, is_persistent=True))
    collection = client.get_or_create_collection(name="chatbot_responses", embedding_function=embedding_function)
    print(f"ChromaDB collection 'chatbot_responses' created or retrieved successfully at {database_path}.")
except Exception as e:
    print(f"An error occurred while setting up persistent ChromaDB: {e}")
    exit()

# In-memory cache for exact matches. This is a faster first-tier cache.
# This cache will be cleared every time the script is restarted.
EXACT_MATCH_CACHE = {}

# --- Functions for Caching ---

def get_semantic_cached_response(query_text, similarity_threshold=0.9):
    """
    Searches the ChromaDB collection for a semantically similar query and returns the cached response.
    """
    # Search the collection for the most similar document
    results = collection.query(
        query_texts=[query_text],
        n_results=1
    )
    
    if not results or not results['ids'] or not results['ids'][0]:
        # print("DEBUG: ChromaDB query returned no semantic results.") # Optional: uncomment for more detailed debug
        return None
            
    distance = results['distances'][0][0]
    cached_document = results['documents'][0][0]
    
    # Check if the calculated distance is below the threshold
    # A smaller distance means higher similarity.
    if distance <= (1 - similarity_threshold):
        print(f"DEBUG: Semantic cache hit! Distance: {distance:.4f} (Threshold: {(1 - similarity_threshold):.4f})")
        return cached_document
    else:
        # print(f"DEBUG: Semantic cache miss. Distance {distance:.4f} is above threshold {(1 - similarity_threshold):.4f}.") # Optional: uncomment for more detailed debug
        return None

def store_response(query_text, bot_response):
    """
    Stores a new query and its response in both the in-memory exact-match cache and ChromaDB for semantic search.
    """
    # Store in the exact-match cache (Tier 1)
    EXACT_MATCH_CACHE[query_text] = bot_response
    print("DEBUG: Response stored in in-memory exact match cache.")
    
    # Store in ChromaDB for semantic search (Tier 2)
    try:
        # Generate a unique ID for the document in ChromaDB
        # In a real application, consider using UUIDs for more robust unique IDs
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

# --- Chatbot Logic ---

def chatbot():
    print("\nWelcome to the Bank Account Opening Chatbot!")
    print("You can start by asking a general question or by saying 'I want to open an account'.")
    
    # Initialize the generative model
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        # --- Caching Logic ---
        # Tier 1 Cache: Check for an exact match in the in-memory dictionary
        if user_input in EXACT_MATCH_CACHE:
            print(f"Bot (from exact cache): {EXACT_MATCH_CACHE[user_input]}")
            continue # Skip to the next iteration of the loop, no need to hit Gemini or ChromaDB
        
        # Tier 2 Cache: If no exact match, perform a semantic search in ChromaDB
        semantic_cached_response = get_semantic_cached_response(user_input)
        
        if semantic_cached_response:
            print(f"Bot (from semantic cache): {semantic_cached_response}")
            # Optionally, add this to the exact cache for faster retrieval next time
            EXACT_MATCH_CACHE[user_input] = semantic_cached_response
        else:
            # If no cache hit at all (neither exact nor semantic), call the Gemini API
            try:
                print("DEBUG: Calling Gemini API for a new response...")
                response = model.generate_content(user_input)
                new_response = response.text
                print(f"Bot (from Gemini): {new_response}")
                
                # Store the new response in both caches for future use
                store_response(user_input, new_response)
                
            except Exception as e:
                print(f"An error occurred while calling the Gemini API: {e}")
                # If Gemini API fails, you might want to provide a fallback message
                print("Bot: I'm sorry, I'm having trouble generating a response right now. Please try again later.")

if __name__ == "__main__":
    chatbot()