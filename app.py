# app.py

import os
import pickle
import faiss
import logging
import re
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
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

# Configure maximum upload size (e.g., 16 MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 Megabytes

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the GROQ and Google API keys
groq_api_key = os.getenv('GROQ_API_KEY')
google_api_key = os.getenv("GOOGLE_API_KEY")

if not groq_api_key or not google_api_key:
    logger.error("API keys for GROQ or Google are missing.")
    raise ValueError("API keys for GROQ or Google are not set in environment variables.")

os.environ["GOOGLE_API_KEY"] = google_api_key

# Initialize ChatGroq with reduced temperature for more accurate policy responses
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192", temperature=0.2)

# Create prompt template with bold headers instead of Markdown headings
prompt = ChatPromptTemplate.from_template(
    """
    You are a highly knowledgeable HR policy expert at a university. Your job is to provide clear, accurate, and comprehensive answers about university policies based on the information in the provided context. 

    <guidelines>
    - Focus exclusively on information contained in the context below.
    - Structure your answers with bold text for main sections and subsections using **Section Title** format.
    - Use bullet points for listing requirements, eligibility criteria, procedures, or steps.
    - Include relevant details from the context about the policy in question.
    - When explaining procedures, present them in chronological order or by importance.
    - Include specific policy numbers, form names, or document references if they appear in the context.
    - Quote exact eligibility criteria, time periods, and deadlines from the context.
    - If the policy mentions exceptions or special cases, include these.
    - For any policy that requires approvals, state who must approve and what the process entails.
    - If multiple sections of the context mention the same policy, synthesize the information.
    - If citing information, note the page number if available (e.g., "According to Page 4 of the policy...").
    - If you cannot find sufficient information about the specific policy asked, simply provide information about related policies that are mentioned in the context without stating limitations or what is missing.
    - Do not include a "Limitations" section or any statements about information not being available.
    - Never suggest contacting HR or other departments for more information.
    </guidelines>

    <context>
    {context}
    </context>

    Question: {input}

    Begin your response with a clear definition or overview of the policy in bold (e.g., **Policy Overview**). Then provide a comprehensive explanation of aspects of the policy mentioned in the context, using bold text for headings and subheadings (e.g., **Eligibility Criteria**, **Application Process**, etc.) and formatting for readability. Keep your response focused on providing helpful information that IS available in the context and avoid mentioning what is NOT available.
    """
)

# Paths for storage
PDF_FOLDER = "shared_storage/pdfs"
VECTOR_FOLDER = "shared_storage/vectors"
VECTORS_FILE = os.path.join(VECTOR_FOLDER, "vectors.index")
DOCUMENTS_FILE = os.path.join(VECTOR_FOLDER, "documents.pkl")

# Ensure necessary directories exist
os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)
os.makedirs("shared_storage", exist_ok=True)

# Define the embedding dimension manually
EMBEDDING_DIMENSION = 768  # As per your model's specification

# Initialize or load existing FAISS index and documents
if os.path.exists(VECTORS_FILE) and os.path.exists(DOCUMENTS_FILE):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # Load existing FAISS index
        index = faiss.read_index(VECTORS_FILE)
        # Load existing documents
        with open(DOCUMENTS_FILE, "rb") as f:
            documents = pickle.load(f)
        # Initialize FAISS vector store
        docstore = InMemoryDocstore({i: doc for i, doc in enumerate(documents)})
        index_to_docstore_id = {i: i for i in range(len(documents))}
        vectors = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )
        logger.info("Loaded existing FAISS index and documents.")
    except Exception as e:
        logger.error(f"Error loading FAISS index and documents: {e}", exc_info=True)
        # Initialize a new FAISS index with empty data
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        docstore = InMemoryDocstore({})
        vectors = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id={}
        )
        documents = []
        logger.info("Initialized new FAISS index and documents due to loading error.")
else:
    # Initialize a new FAISS index with empty data
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
    docstore = InMemoryDocstore({})
    vectors = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id={}
    )
    documents = []
    logger.info("Initialized new FAISS index and documents.")

# Function to extract policy number from text if available
def extract_policy_number(text):
    """Extract policy number from text using regex pattern matching."""
    # Common patterns for policy numbers: Policy X.X, Policy X.X.X, etc.
    patterns = [
        r'Policy\s+(\d+\.\d+(?:\.\d+)?)',  # Policy X.X or Policy X.X.X
        r'Policy\s+No\.\s*(\d+\.\d+(?:\.\d+)?)',  # Policy No. X.X
        r'Policy\s+Number\s*(\d+\.\d+(?:\.\d+)?)',  # Policy Number X.X
        r'University\s+Policy\s*(\d+\.\d+(?:\.\d+)?)'  # University Policy X.X
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[0]
    
    return None

# Detect section headers and add them to metadata
def extract_section_headers(text):
    """Extract policy section headers for better context."""
    # Look for common section header patterns
    section_pattern = r'^(?:\d+\.)?\s*([A-Z][A-Za-z\s]{2,50})(?:\:|\.\s|$)'
    headers = []
    
    for line in text.split('\n'):
        line = line.strip()
        match = re.match(section_pattern, line)
        if match and len(line) < 100:  # Avoid matching paragraphs
            header = match.group(1).strip()
            if header and not any(word in header.lower() for word in ['page', 'university', 'policy']):
                headers.append(header)
    
    return headers

# Function to save FAISS index and documents
def save_vector_store():
    """Save the FAISS index and documents to disk."""
    try:
        faiss.write_index(vectors.index, VECTORS_FILE)
        with open(DOCUMENTS_FILE, "wb") as f:
            pickle.dump(documents, f)
        logger.info("Saved FAISS index and documents.")
    except Exception as e:
        logger.error(f"Error saving FAISS index and documents: {e}", exc_info=True)

# Improved PDF processing function with better text extraction and metadata
def process_pdf(file_path):
    """Extract text from a PDF, split it, embed, and add to the vector store."""
    try:
        pdf_reader = PdfReader(file_path)
        total_pages = len(pdf_reader.pages)
        filename = os.path.basename(file_path)
        policy_number = None
        full_text = ""
        
        # First pass: Extract full text and look for policy number
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            try:
                extracted_text = page.extract_text()
                if extracted_text:
                    full_text += extracted_text + " "
                    # Try to find policy number on first few pages
                    if page_num <= 3 and not policy_number:
                        policy_number = extract_policy_number(extracted_text)
            except Exception as e:
                logger.error(f"First-pass error extracting text from page {page_num}: {e}")
        
        # Extract any section headers from the full document
        section_headers = extract_section_headers(full_text)
        
        # Second pass: Process each page with clear page markers
        text = ""
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            try:
                extracted_text = page.extract_text()
                if extracted_text:
                    # Format: [PAGE X OF Y] for easy citation in responses
                    text += f"[PAGE {page_num} OF {total_pages}]\n{extracted_text}\n\n"
                else:
                    logger.warning(f"No text found on page {page_num} of {file_path}")
            except Exception as page_e:
                logger.error(f"Error extracting text from page {page_num} of {file_path}: {page_e}")
        
        if not text.strip():
            logger.warning(f"No text extracted from PDF: {file_path}")
            return
        
        # Add detailed metadata for HR policy document
        doc = Document(
            page_content=text, 
            metadata={
                "source": file_path,
                "filename": filename,
                "total_pages": total_pages,
                "type": "policy_document",
                "policy_number": policy_number,
                "section_headers": section_headers if section_headers else None
            }
        )
        
        # Optimize chunking for policy documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,  # Smaller chunks for more precise retrieval
            chunk_overlap=250,  # Higher overlap to maintain policy context
            separators=["\n\n", "\n", ". ", " ", ""]  # Custom separators for policy documents
        )
        
        chunks = text_splitter.split_documents([doc])
        
        # Add detailed metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_count"] = len(chunks)
            # Look for page markers in the chunk text to set page numbers
            page_markers = re.findall(r'\[PAGE (\d+) OF \d+\]', chunk.page_content)
            if page_markers:
                chunk.metadata["page_numbers"] = page_markers
            
        vectors.add_documents(chunks)
        documents.extend(chunks)
        save_vector_store()
        logger.info(f"Processed and embedded PDF: {file_path} - Created {len(chunks)} chunks")
        if policy_number:
            logger.info(f"Identified Policy Number: {policy_number}")
        if section_headers:
            logger.info(f"Identified Section Headers: {', '.join(section_headers[:5])}...")
    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {e}", exc_info=True)

# Function to remove a PDF from the vector store
def remove_pdf_from_vector_store(file_path):
    """Remove all documents associated with a PDF from the vector store."""
    try:
        global documents, vectors
        # Identify documents to remove
        documents_to_remove = [doc for doc in documents if doc.metadata.get('source') == file_path]
        if not documents_to_remove:
            logger.warning(f"No documents found for PDF: {file_path}")
            return
        
        # Remove documents
        documents = [doc for doc in documents if doc.metadata.get('source') != file_path]
        
        # Recreate the FAISS vector store
        vectors = FAISS.from_documents(documents, embeddings)
        save_vector_store()
        logger.info(f"Removed PDF from vector store: {file_path}")
    except Exception as e:
        logger.error(f"Error removing PDF from vector store {file_path}: {e}", exc_info=True)


# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to get metadata (e.g., version info)
@app.route('/meta.json')
def meta():
    metadata = {
        "version": "1.0.0",
        "description": "University HR Policy Chatbot"
    }
    return jsonify(metadata)

# Route to manage PDFs (Upload and List)
@app.route('/manage_pdfs', methods=['GET', 'POST'])
def manage_pdfs():
    error_message = None
    uploaded_files_history = []
    if request.method == 'POST':
        uploaded_files = request.files.getlist('pdf')
        if not uploaded_files:
            error_message = "No files selected for upload."
            return render_template('manage_pdfs.html', error_message=error_message, uploaded_files_history=uploaded_files_history)
        
        for uploaded_file in uploaded_files:
            if uploaded_file and allowed_file(uploaded_file.filename):
                filename = secure_filename(uploaded_file.filename)
                file_path = os.path.join(PDF_FOLDER, filename)
                if os.path.exists(file_path):
                    error_message = f"File already exists: {filename}"
                    continue
                uploaded_file.save(file_path)
                process_pdf(file_path)
            else:
                error_message = f"Invalid file type: {uploaded_file.filename}"
                continue
        return redirect(url_for('manage_pdfs'))
    
    # For GET requests, list all uploaded PDFs
    try:
        uploaded_files = os.listdir(PDF_FOLDER)
        uploaded_files = [f for f in uploaded_files if allowed_file(f)]
        for file_name in uploaded_files:
            uploaded_files_history.append({
                'file_name': file_name
            })
    except Exception as e:
        logger.error(f"Error listing uploaded PDFs: {e}", exc_info=True)
        error_message = "Error retrieving uploaded PDFs."
    
    return render_template('manage_pdfs.html', error_message=error_message, uploaded_files_history=uploaded_files_history)

# Route to handle PDF actions: rename, update, delete
@app.route('/pdf_action', methods=['POST'])
def pdf_action():
    action = request.form.get('action')
    file_name = request.form.get('file_name')
    new_name = request.form.get('new_name')
    
    logger.info(f"Received action: {action} for file: {file_name}")

    if not file_name:
        logger.error("PDF action failed: File name is missing.")
        return jsonify({"error": "File name is missing."}), 400
    
    file_path = os.path.join(PDF_FOLDER, file_name)
    
    if not os.path.exists(file_path):
        logger.error(f"PDF action failed: File not found - {file_path}")
        return jsonify({"error": "File not found."}), 404
    
    if action == 'rename':
        if not new_name:
            logger.error("Rename action failed: New name is missing.")
            return jsonify({"error": "New name is missing."}), 400
        # Ensure the new filename is secure and ends with .pdf
        new_filename = secure_filename(new_name) + '.pdf'
        new_file_path = os.path.join(PDF_FOLDER, new_filename)
        if os.path.exists(new_file_path):
            logger.error(f"Rename action failed: A file with the new name already exists - {new_filename}")
            return jsonify({"error": "A file with the new name already exists."}), 400
        try:
            os.rename(file_path, new_file_path)
            # Update vector store
            remove_pdf_from_vector_store(file_path)
            process_pdf(new_file_path)
            logger.info(f"Renamed PDF from {file_name} to {new_filename}")
            return jsonify({"message": "File renamed successfully."}), 200
        except Exception as e:
            logger.error(f"Rename action failed for {file_name}: {e}", exc_info=True)
            return jsonify({"error": "Failed to rename the file."}), 500
    
    elif action == 'update':
        if 'new_file' not in request.files:
            logger.error("Update action failed: No file part in the request.")
            return jsonify({"error": "No file part."}), 400
        new_file = request.files['new_file']
        if new_file.filename == '':
            logger.error("Update action failed: No file selected for upload.")
            return jsonify({"error": "No selected file."}), 400
        if new_file and allowed_file(new_file.filename):
            # Replace the existing file
            new_filename = secure_filename(new_file.filename)
            new_file_path = os.path.join(PDF_FOLDER, new_filename)
            if new_filename != file_name and os.path.exists(new_file_path):
                logger.error(f"Update action failed: A file with the new name already exists - {new_filename}")
                return jsonify({"error": "A file with the new name already exists."}), 400
            try:
                new_file.save(new_file_path)
                # Remove old PDF from vector store
                remove_pdf_from_vector_store(file_path)
                # If filename changed, remove old file
                if new_filename != file_name:
                    os.remove(file_path)
                # Process the new PDF
                process_pdf(new_file_path)
                logger.info(f"Updated PDF: {file_name} with new file: {new_filename}")
                return jsonify({"message": "File updated successfully."}), 200
            except Exception as e:
                logger.error(f"Update action failed for {file_name}: {e}", exc_info=True)
                return jsonify({"error": "Failed to update the file."}), 500
        else:
            logger.error("Update action failed: Invalid file type.")
            return jsonify({"error": "Invalid file type."}), 400
    
    elif action == 'delete':
        try:
            os.remove(file_path)
            remove_pdf_from_vector_store(file_path)
            logger.info(f"Deleted PDF: {file_name}")
            return jsonify({"message": "File deleted successfully."}), 200
        except Exception as e:
            logger.error(f"Delete action failed for {file_name}: {e}", exc_info=True)
            return jsonify({"error": "Error deleting file."}), 500
    else:
        logger.error(f"Invalid action received: {action}")
        return jsonify({"error": "Invalid action."}), 400

# Route for the Ask Question page
@app.route('/ask_question', methods=['GET', 'POST'])
def ask_question():
    if request.method == 'POST':
        # Handle form data
        question = request.form.get('question')
        
        if not question:
            logger.error("Ask Question action failed: Question is missing.")
            return jsonify({"error": "Question is missing."}), 400
        
        try:
            if not documents:
                logger.error("Ask Question action failed: No documents available.")
                return jsonify({"error": "No documents available for answering questions."}), 400
            
            # Create an in-memory docstore
            docstore = InMemoryDocstore({i: doc for i, doc in enumerate(documents)})
            vectors_local = FAISS(
                embedding_function=embeddings,
                index=vectors.index,
                docstore=docstore,
                index_to_docstore_id={i: i for i in range(len(documents))}
            )

            # Enhanced retriever optimized for policy questions - using similarity instead of MMR
            # for better focus on the most relevant chunks
            retriever = vectors_local.as_retriever(
                search_type="similarity",  # Use similarity for most relevant results
                search_kwargs={
                    "k": 12,  # Retrieve more chunks to find any related information
                    "fetch_k": 30  # Consider even more candidates before selecting the final k
                }
            )

            # Prepare and retrieve context for the question
            document_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            # Get the answer
            response = retrieval_chain.invoke({'input': question})
            answer = response['answer']

            # Provide detailed logging for debugging
            logger.info(f"Question: {question}, Answer generated successfully")
            # Avoid logging full answer text to keep logs manageable
            logger.debug(f"Answer first 100 chars: {answer[:100]}...")

            return jsonify({"answer": answer, "progress": 1.0})  # Assuming progress is complete
        except Exception as e:
            logger.error(f"Error processing question: {question}. Error: {e}", exc_info=True)
            return jsonify({"error": "An error occurred while processing your question."}), 500
    
    # For GET requests, render the page
    return render_template('ask_question.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)