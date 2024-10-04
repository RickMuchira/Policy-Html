import os
import pickle
import hashlib
import faiss
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.docstore.in_memory import InMemoryDocstore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the GROQ and Google API keys
groq_api_key = os.getenv('GROQ_API_KEY')
google_api_key = os.getenv("GOOGLE_API_KEY")

if not groq_api_key or not google_api_key:
    logger.error("API keys for GROQ or Google are missing.")
    raise ValueError("API keys for GROQ or Google are not set in environment variables.")

os.environ["GOOGLE_API_KEY"] = google_api_key

# Initialize ChatGroq
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Create ChatPromptTemplate
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Load previously uploaded files
uploaded_files_history = []
if os.path.exists("shared_storage/uploaded_files_history.pkl"):
    with open("shared_storage/uploaded_files_history.pkl", "rb") as f:
        uploaded_files_history = pickle.load(f)

# Function to perform vector embedding with batching
def embed_pdfs(docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = []

    for doc in docs:
        try:
            chunks = text_splitter.split_documents([doc])
            final_documents.extend(chunks)
        except Exception as e:
            logger.error(f"Error splitting document: {doc.metadata['source']}. Error: {e}")
            continue

    vectors = None
    batch_size = 100  # API batch size limit
    for i in range(0, len(final_documents), batch_size):
        batch_documents = final_documents[i:i+batch_size]
        try:
            batch_vectors = FAISS.from_documents(batch_documents, embeddings)
            if vectors is None:
                vectors = batch_vectors
            else:
                vectors.index.add(batch_vectors.index.reconstruct_n(0, len(batch_documents)))
        except Exception as e:
            logger.error(f"Error embedding batch {i//batch_size + 1}: {e}")
            continue

    return vectors, final_documents

# Function to delete a PDF
def delete_pdf(file_name):
    global uploaded_files_history

    try:
        uploaded_files_history = [file for file in uploaded_files_history if file != file_name]
        with open("shared_storage/uploaded_files_history.pkl", "wb") as f:
            pickle.dump(uploaded_files_history, f)

        with open("shared_storage/documents.pkl", "rb") as f:
            documents = pickle.load(f)

        documents = [doc for doc in documents if doc.metadata["source"] != file_name]
        with open("shared_storage/documents.pkl", "wb") as f:
            pickle.dump(documents, f)

        vectors, _ = embed_pdfs(documents)
        faiss.write_index(vectors.index, "shared_storage/vectors.index")
    except Exception as e:
        logger.error(f"Error deleting PDF: {file_name}. Error: {e}")

# Function to rename a PDF
def rename_pdf(old_name, new_name):
    global uploaded_files_history

    try:
        uploaded_files_history = [new_name if file == old_name else file for file in uploaded_files_history]
        with open("shared_storage/uploaded_files_history.pkl", "wb") as f:
            pickle.dump(uploaded_files_history, f)

        with open("shared_storage/documents.pkl", "rb") as f:
            documents = pickle.load(f)

        for doc in documents:
            if doc.metadata["source"] == old_name:
                doc.metadata["source"] = new_name

        with open("shared_storage/documents.pkl", "wb") as f:
            pickle.dump(documents, f)

        vectors, _ = embed_pdfs(documents)
        faiss.write_index(vectors.index, "shared_storage/vectors.index")
    except Exception as e:
        logger.error(f"Error renaming PDF from {old_name} to {new_name}. Error: {e}")

# Function to update a PDF
def update_pdf(old_name, new_file):
    global uploaded_files_history

    try:
        uploaded_files_history = [new_file.filename if file == old_name else file for file in uploaded_files_history]
        with open("shared_storage/uploaded_files_history.pkl", "wb") as f:
            pickle.dump(uploaded_files_history, f)

        with open("shared_storage/documents.pkl", "rb") as f:
            documents = pickle.load(f)

        documents = [doc for doc in documents if doc.metadata["source"] != old_name]
        pdf_reader = PdfReader(new_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        new_doc = Document(page_content=text, metadata={"source": new_file.filename})
        documents.append(new_doc)

        with open("shared_storage/documents.pkl", "wb") as f:
            pickle.dump(documents, f)

        vectors, _ = embed_pdfs(documents)
        faiss.write_index(vectors.index, "shared_storage/vectors.index")
    except Exception as e:
        logger.error(f"Error updating PDF: {old_name}. Error: {e}")

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for managing PDFs
@app.route('/manage-pdfs', methods=['GET', 'POST'])
def manage_pdfs():
    if request.method == 'POST':
        uploaded_files = request.files.getlist("pdf")
        all_docs = []
        new_files = []

        for uploaded_file in uploaded_files:
            try:
                pdf_reader = PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()

                doc = Document(page_content=text, metadata={"source": uploaded_file.filename})
                all_docs.append(doc)
                new_files.append(uploaded_file.filename)
            except Exception as e:
                logger.error(f"Error processing uploaded file: {uploaded_file.filename}. Error: {e}")
                continue

        vectors, final_documents = embed_pdfs(all_docs)

        if not os.path.exists("shared_storage"):
            os.makedirs("shared_storage")

        faiss.write_index(vectors.index, "shared_storage/vectors.index")
        with open("shared_storage/documents.pkl", "wb") as f:
            pickle.dump(final_documents, f)

        uploaded_files_history.extend(new_files)
        with open("shared_storage/uploaded_files_history.pkl", "wb") as f:
            pickle.dump(uploaded_files_history, f)

        return redirect(url_for('manage_pdfs'))
    
    search_query = request.args.get('search_query', '')
    filtered_files = [file for file in uploaded_files_history if search_query.lower() in file.lower()]

    return render_template('manage_pdfs.html', uploaded_files_history=filtered_files)

# Route for deleting, renaming, or updating PDFs
@app.route('/pdf-action', methods=['POST'])
def pdf_action():
    action = request.form['action']
    file_name = request.form['file_name']

    if action == 'delete':
        delete_pdf(file_name)
    elif action == 'rename':
        new_name = request.form['new_name']
        if new_name:
            rename_pdf(file_name, new_name)
    elif action == 'update':
        new_file = request.files['new_file']
        if new_file:
            update_pdf(file_name, new_file)

    # After any action, ensure all PDFs are embedded in the knowledge base
    try:
        all_docs = []
        for file_name in uploaded_files_history:
            with open(os.path.join("path/to/uploaded/files", file_name), "rb") as f:
                pdf_reader = PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                doc = Document(page_content=text, metadata={"source": file_name})
                all_docs.append(doc)

        vectors, final_documents = embed_pdfs(all_docs)
        faiss.write_index(vectors.index, "shared_storage/vectors.index")
        with open("shared_storage/documents.pkl", "wb") as f:
            pickle.dump(final_documents, f)
    except Exception as e:
        logger.error(f"Error re-embedding all documents after action. Error: {e}")

    return redirect(url_for('manage_pdfs'))

# Route for asking questions (AJAX route for chatbot interaction)
@app.route('/ask-question', methods=['POST'])
def ask_question():
    question = request.json.get('question', '')

    if not question:
        return jsonify({"error": "No question provided."}), 400

    try:
        # Retrieve FAISS index and documents
        index = faiss.read_index("shared_storage/vectors.index")
        with open("shared_storage/documents.pkl", "rb") as f:
            documents = pickle.load(f)

        # Create an in-memory docstore
        docstore = InMemoryDocstore({i: doc for i, doc in enumerate(documents)})
        vectors = FAISS(embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"), 
                        docstore=docstore, 
                        index=index, 
                        index_to_docstore_id={i: i for i in range(len(documents))})

        # Prepare and retrieve context for the question
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Get the answer
        response = retrieval_chain.invoke({'input': question})
        answer = response['answer']

        # Provide detailed logging for debugging
        logger.info(f"Question: {question}, Answer: {answer}")
        
        return jsonify({"answer": answer})
    except Exception as e:
        logger.error(f"Error processing question: {question}. Error: {e}")
        return jsonify({"error": "An error occurred while processing your question."}), 500

if __name__ == '__main__':
    app.run(debug=True)
