# whatsapp.py

import os
import pickle
import logging
from flask import Flask, request, abort
from langchain_text_splitters import RecursiveCharacterTextSplitter
from twilio.twiml.messaging_response import MessagingResponse
import faiss
from langchain.docstore.document import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.docstore.in_memory import InMemoryDocstore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the GROQ and Google API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize ChatGroq
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Create ChatPromptTemplate
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

# Initialize Flask app
whatsapp_app = Flask(__name__)

# Function to perform vector embedding
def embed_pdfs(docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = []

    for doc in docs:
        chunks = text_splitter.split_documents([doc])
        final_documents.extend(chunks)

    vectors = FAISS.from_documents(final_documents, embeddings)
    return vectors, final_documents

# Function to generate response
def generate_response(question, vectors):
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({'input': question})
    return response['answer']

# Route to handle incoming WhatsApp messages
@whatsapp_app.route('/whatsapp', methods=['POST'])
def whatsapp_receive():
    incoming_msg = request.values.get('Body', '').strip()

    try:
        # Send an opening message if the incoming message is the first interaction
        if incoming_msg.lower() == 'hi' or incoming_msg.lower() == 'hello':
            opening_message = (
                "Hello! Welcome to our WhatsApp service. "
                "You can ask me questions about the People and Culture Policy Manual . "
                "Simply type your question and I will provide you with the best possible answer."
            )
            resp = MessagingResponse()
            resp.message(opening_message)
            return str(resp)

        if not incoming_msg:
            abort(400, "No message content found.")

        index = faiss.read_index("shared_storage/vectors.index")
        with open("shared_storage/documents.pkl", "rb") as f:
            documents = pickle.load(f)

        docstore = InMemoryDocstore({i: doc for i, doc in enumerate(documents)})
        vectors = FAISS(
            embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
            docstore=docstore,
            index=index,
            index_to_docstore_id={i: i for i in range(len(documents))}
        )

        answer = generate_response(incoming_msg, vectors)

        resp = MessagingResponse()
        resp.message(answer)
        return str(resp)

    except Exception as e:
        logger.error(f"Error processing message: {incoming_msg}. Error: {e}")
        abort(500, "An error occurred while processing your message.")

if __name__ == '__main__':
    whatsapp_app.run(debug=True, port=5001)
