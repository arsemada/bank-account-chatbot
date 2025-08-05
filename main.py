import os
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
from chromadb.config import Settings # Import Settings for persistent client

# --- Configuration and Initialization ---

# Load environment variables
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

# --- New Functions for Caching ---

def get_cached_response(query_text, similarity_threshold=0.9):
    """
    Searches the ChromaDB collection for a similar query and returns the cached response.
    """
    # Search the collection for the most similar document
    results = collection.query(
        query_texts=[query_text],
        n_results=1
    )
    
    # Check if a result was found and if its similarity score is above the threshold
    # ChromaDB returns distances. A smaller distance means higher similarity.
    # The distance is typically 1 - cosine_similarity. So, a threshold of 0.9 similarity
    # means a distance <= 0.1.
    if results and results['ids'] and results['ids'][0]: # Check if any results were returned
        distance = results['distances'][0][0]
        if distance <= (1 - similarity_threshold):
            cached_response = results['documents'][0][0]
            print(f"DEBUG: Found cached response with distance {distance:.4f} (Similarity: {1-distance:.4f})")
            return cached_response
        
    print("DEBUG: No sufficiently similar response found in cache.")
    return None

def store_response(query_text, bot_response):
    """
    Stores a new query and its response in the ChromaDB collection.
    """
    # ChromaDB requires a unique ID for each document.
    # We'll use a simple counter based on the current number of documents in the collection.
    # In a real-world app, you might use UUIDs or a more robust ID generation.
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
        
        # 1. Check for a cached response
        cached_response = get_cached_response(user_input)
        
        if cached_response:
            # 2. If a cached response is found, return it
            print(f"Bot (from cache): {cached_response}")
        else:
            # 3. If not, call the Gemini API for a new response
            try:
                response = model.generate_content(user_input)
                new_response = response.text
                print(f"Bot (from Gemini): {new_response}")
                
                # 4. Store the new response in ChromaDB
                store_response(user_input, new_response)
                
            except Exception as e:
                print(f"An error occurred while calling the Gemini API: {e}")

if __name__ == "__main__":
    chatbot()