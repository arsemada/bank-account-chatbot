# Bank Account Chatbot 🏦

This is a modern, AI-powered chatbot built to assist bank customers with their inquiries. The application provides two main functionalities: a general-purpose chat to answer banking questions and a guided workflow for opening a new account. The project leverages **Google's Gemini API** for natural language understanding, **ChromaDB** for efficient caching, and **Streamlit** for an interactive web interface.

### Key Features

* **Interactive and Modern UI:** The user interface is built with Streamlit and features a custom dark theme, a clean chat history display, and a functional sidebar. This provides a user-friendly and aesthetically pleasing experience.

* **Stateful Conversation Flow:** The chatbot can guide users through a multi-step process, such as opening a new account, by remembering the context of the conversation. This ensures a smooth and logical experience for complex tasks.

* **Intelligent Caching System:** To optimize performance and reduce API costs, the chatbot uses a two-tiered caching system:
    * An in-memory exact-match cache for immediate, identical queries.
    * A persistent **ChromaDB** vector store for semantic search, allowing the bot to find answers to similar, but not identical, questions.

* **Quick Answers (FAQ Section):** The sidebar includes a dedicated FAQ section with collapsible expanders. This allows users to get immediate answers to common questions without needing to engage the AI, making the chatbot more efficient.

* **New Chat Functionality:** A "New Chat" button in the sidebar gives the user full control to clear the conversation history and start a new session, ensuring a clean and focused interaction.
