import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure the Gemini API
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Gemini API configured successfully.")
else:
    print("Error: GOOGLE_API_KEY not found in environment variables.")
    exit()

# Now, let's test the connection by getting a list of available models
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print(f"An error occurred while listing models: {e}")

# Your chatbot logic will go here
def chatbot():
    print("Welcome to the Bank Account Opening Chatbot!")
    print("You can start by asking a general question or by saying 'I want to open an account'.")

    # This is where we will add our main loop later
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        # For now, we'll just echo the input
        print(f"Bot (for now): You said '{user_input}'")

if __name__ == "__main__":
    chatbot()