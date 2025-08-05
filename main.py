import os
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from chromadb.utils import embedding_functions

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
# We are using the 'models/embedding-001' model
# Note: The model name in the Gemini API is 'models/text-embedding-004' but ChromaDB uses 'models/embedding-001' by default.
# It's important to be consistent. Let's stick with 'models/embedding-001' for now.
embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=GOOGLE_API_KEY, model_name="models/embedding-001")

# Initialize the ChromaDB client
# By default, this creates a local, in-memory database.
client = chromadb.Client()

# Create or get a collection. The collection is where we'll store our documents.
try:
    collection = client.get_or_create_collection(name="chatbot_responses", embedding_function=embedding_function)
    print("ChromaDB collection 'chatbot_responses' created or retrieved successfully.")
except Exception as e:
    print(f"An error occurred while setting up ChromaDB: {e}")
    exit()

# --- Chatbot Logic ---

# Your chatbot logic will go here
def chatbot():
    print("\nWelcome to the Bank Account Opening Chatbot!")
    print("You can start by asking a general question or by saying 'I want to open an account'.")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # For now, we'll just echo the input
        print(f"Bot (for now): You said '{user_input}'")

if __name__ == "__main__":
    chatbot()