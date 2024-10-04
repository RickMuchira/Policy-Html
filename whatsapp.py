import os
import logging
from flask import Flask, request, abort
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from dotenv import load_dotenv

# Import functions from app.py
from app import load_faiss_index, generate_response

# Load environment variables from .env
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Twilio credentials and number from environment variables
twilio_account_sid = os.getenv('TWILIO_ACCOUNT_SID')
twilio_auth_token = os.getenv('TWILIO_AUTH_TOKEN')
twilio_number = os.getenv('TWILIO_PHONE_NUMBER')  # WhatsApp-enabled Twilio number

# Initialize Twilio client
twilio_client = Client(twilio_account_sid, twilio_auth_token)

# Initialize Flask app
whatsapp_app = Flask(__name__)

# Route to handle incoming WhatsApp messages
@whatsapp_app.route('/whatsapp', methods=['POST'])
def whatsapp_receive():
    # Get the message content and sender's phone number from the request
    incoming_msg = request.values.get('Body', '').strip()
    sender_number = request.values.get('From', '').strip()

    logger.info(f"Received message from {sender_number}: {incoming_msg}")

    try:
        # Handle greetings
        if incoming_msg.lower() in ['hi', 'hello']:
            opening_message = (
                "Hello! Welcome to our WhatsApp service. "
                "You can ask me questions about the People and Culture Policy Manual. "
                "Simply type your question and I will provide you with the best possible answer."
            )
            resp = MessagingResponse()
            resp.message(opening_message)
            return str(resp)

        if not incoming_msg:
            abort(400, "No message content found.")

        # Use functions from app.py to load FAISS index and generate a response
        vectors = load_faiss_index()  # Load FAISS index and documents
        answer = generate_response(incoming_msg, vectors)  # Generate the response based on the message

        # Send the response to the user via WhatsApp
        resp = MessagingResponse()
        resp.message(answer)
        return str(resp)

    except Exception as e:
        logger.error(f"Error processing message: {incoming_msg}. Error: {e}")
        abort(500, "An error occurred while processing your message.")


if __name__ == '__main__':
    # Start the Flask app for WhatsApp on port 5001
    whatsapp_app.run(debug=True, port=5001)
