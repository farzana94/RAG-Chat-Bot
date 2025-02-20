from flask import Flask, request, render_template, jsonify
from langchain_community.document_loaders import PyPDFLoader,  TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_retrieval_chain
import os
import asyncio

app = Flask(__name__)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Global variables
chat_retrieval_chain = None
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True) 
FAISS_INDEX_PATH = "faiss_index"

# Initialize Hugging Face Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load or create FAISS index
if os.path.exists(FAISS_INDEX_PATH):
    db = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
else:
    db = None  # No FAISS index exists yet

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
async def upload_file():
    global chat_retrieval_chain, db

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    file_extension = file.filename.split('.')[-1].lower()
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    # Choose the correct loader
    if file_extension == 'pdf':
        loader = PyPDFLoader(file_path)
    elif file_extension == 'txt':
        loader = TextLoader(file_path, encoding="utf-8", autodetect_encoding=True)
    elif file_extension == 'docx':
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        return jsonify({'error': 'Unsupported file format. Only PDF, TXT, and DOCX are allowed.'})
    try:
        docs = await asyncio.to_thread(loader.load)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        documents = await asyncio.to_thread(text_splitter.split_documents, docs)
        # Assign source metadata
        # Normalize source metadata (remove full path)
        for doc in documents:
            doc.metadata["source"] = os.path.basename(file.filename)
        # Initialize or update FAISS index
        if db is None:
            db = await asyncio.to_thread(FAISS.from_documents, documents, embedding_model)
        else:
            db.add_documents(documents)

        # Save FAISS index
        await asyncio.to_thread(db.save_local, FAISS_INDEX_PATH)

        retriever = db.as_retriever(search_kwargs={"k": 5})

        # Use Gemini Flash API as LLM
        llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

        prompt = ChatPromptTemplate.from_template("""
        Answer the following question based only on the provided context.
        Think step by step before providing a detailed answer.
        <context>
        {context}
        </context>
        Question: {input}""")

        document_chain = create_stuff_documents_chain(llm, prompt)
        chat_retrieval_chain = create_retrieval_chain(retriever, document_chain)

        return jsonify({'success': 'File uploaded and processed successfully'})
    except Exception as e:
        return jsonify({'error': f'Failed to process file: {str(e)}'})

@app.route('/chat', methods=['POST'])
async def chat():
    global chat_retrieval_chain
    if chat_retrieval_chain is None:
        return jsonify({'error': 'No document uploaded yet. Please upload a PDF first.'})

    user_query = request.json.get('query')
    if not user_query:
        return jsonify({'error': 'No query provided'})
    # Get chat history from memory
    chat_history = memory.load_memory_variables({}).get("chat_history", [])
    response = await asyncio.to_thread(chat_retrieval_chain.invoke, {"input": user_query, "chat_history": chat_history})
    # Extract document sources
    retrieved_docs = response.get('context', [])
    source_files = list(set(doc.metadata.get("source", "Unknown") for doc in retrieved_docs if "source" in doc.metadata))
    # Debugging: Print extracted sources to verify
    print("Extracted Sources:", source_files)
    # Update memory with the new interaction
    memory.save_context({"input": user_query}, {"output": response['answer']})

    return jsonify({
        'answer': response.get("answer", "No response generated."),
        'sources': source_files
    })

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)