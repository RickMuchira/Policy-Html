import os
import logging
from flask import Flask, request, abort, jsonify
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv
from app import ask_question  # Import the question-answering function from app.py

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
whatsapp_app = Flask(__name__)

# Route to handle incoming WhatsApp messages
@whatsapp_app.route('/whatsapp', methods=['POST'])
def whatsapp_receive():
    try:
        # Get the incoming message and sender's phone number
        incoming_msg = request.values.get('Body', '').strip()
        sender_number = request.values.get('From', '').strip()

        logger.info(f"Received message from {sender_number}: {incoming_msg}")

        # Handle greetings
        if incoming_msg.lower() in ['hi', 'hello']:
            opening_message = (
                "Hello! Welcome to our WhatsApp service. "
                "You can ask me questions based on the documents we have. "
                "Simply type your question and I will provide you with the best possible answer."
            )
            resp = MessagingResponse()
            resp.message(opening_message)
            return str(resp)

        # If the message is empty, return an error
        if not incoming_msg:
            logger.error("Empty message received.")
            abort(400, "No message content found.")

        # Process the question using `ask_question` from app.py
        try:
            # Create the input for the `ask_question` function
            question_data = {'question': incoming_msg}

            # Invoke the function and get the answer (using Flask test request context)
            with whatsapp_app.test_request_context('/ask-question', json=question_data):
                answer_response = ask_question()

            # Parse the answer from the Flask response object
            answer_data = answer_response.json
            answer = answer_data.get("answer", "Sorry, I couldn't find an answer to your question.")
        except Exception as e:
            logger.error(f"Error generating answer for message: {incoming_msg}. Error: {e}")
            answer = "Sorry, an error occurred while processing your request."

        # Send the answer back via WhatsApp
        resp = MessagingResponse()
        resp.message(answer)
        return str(resp)

    except Exception as e:
        logger.error(f"Error processing WhatsApp message. Error: {e}")
        abort(500, "An error occurred while processing your message.")

if __name__ == '__main__':
    # Start the Flask app for WhatsApp on port 5001 (or another port as needed)
    whatsapp_app.run(debug=True, port=5001)
