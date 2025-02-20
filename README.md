Here’s a README.md file for your Flask-based RAG chatbot:

Flask-Based RAG Chatbot with FAISS & Gemini Flash

This project is a Retrieval-Augmented Generation (RAG) chatbot using Flask, FAISS, Hugging Face embeddings, and Google Gemini Flash for conversational AI.

Features
	•	📂 Document Uploads: Supports PDF, TXT, and DOCX files.
	•	🔎 Semantic Search: Uses FAISS for efficient retrieval.
	•	🤖 Conversational AI: Powered by Google Gemini Flash API.
	•	💾 Chat Memory: Maintains conversation context.

Installation & Setup

1. Clone the Repository

git clone https://github.com/your-repo/rag-chatbot.git
cd rag-chatbot

2. Create a Virtual Environment

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies

pip install -r requirements.txt

4. Set Up Environment Variables

Create a .env file and add your Google API key:

GOOGLE_API_KEY=your_google_api_key

Usage

Run the Flask App

python app.py

Access the chatbot UI at http://127.0.0.1:5000/.

API Endpoints

1. Upload File

2. Chat
