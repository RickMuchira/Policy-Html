# Policy-Html

This project is a web-based application that allows users to interact with a chatbot for asking questions related to uploaded PDFs. Additionally, there is WhatsApp integration that provides the same functionality via a WhatsApp interface.

## Features

- **PDF Management**: Upload, manage, and update PDFs.
- **Chatbot Interface**: Ask questions related to the uploaded PDFs and receive relevant answers.
- **WhatsApp Integration**: Interact with the chatbot via WhatsApp.
- **Real-time Processing**: Uses Google and GROQ APIs for processing and responding to queries.

## Project Structure


├── app.py                   # Main Flask application for web interaction
├── whatsapp.py              # Flask application for WhatsApp interaction
├── templates/               # HTML templates for the Flask app
│   ├── ask_question.html    # Template for the chatbot question interface
│   ├── index.html           # Home page template
│   ├── manage_pdfs.html     # Template for managing PDFs
├── static/                  # Static files (CSS, JS)
│   ├── style.css            # Styles for the web pages
│   └── script.js            # JavaScript for the chatbot functionality
└── shared_storage/          # Directory for storing uploaded files and embeddings
    ├── uploaded_files_history.pkl
    ├── documents.pkl
    └── vectors.index

## Installation

### Step 1: Clone the Repository
git clone https://github.com/RickMuchira/Policy-Html.git

### Step 2: Create and Activate Virtual Environment
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

## Step 3: Install Dependencies
pip install -r requirements.txt

## Step 4: Set Up Environment Variables
Create a .env file in the root directory with the following content:

- GROQ_API_KEY=your_groq_api_key
- GOOGLE_API_KEY=your_google_api_key

## How to make it work

### Web Application

1. **Run the Flask Application**:
   - Start the web application by running the following command:
     ```bash
     python app.py
     ```
   - This will start the Flask server locally.

2. **Access the Web Application**:
   - Open your web browser and navigate to `http://127.0.0.1:5000/`.
   - You will see the homepage of the application.

3. **Upload PDFs**:
   - Navigate to the "http://127.0.0.1:5000/manage-pdfs" .
   - Use the file upload form to upload one or multiple PDFs. Click the "Upload" button to process the files.
   - The uploaded PDFs will be stored and indexed for later retrieval and querying.

4. **Ask Questions**:
   - Go back to the homepage `http://127.0.0.1:5000/`.
   - Use the chatbot interface to ask questions related to the uploaded PDFs.
   - Type your question in the provided text area and submit it.
   - The chatbot will respond with answers based on the content of the uploaded PDFs.

5. **Manage PDFs**:
   - In the "Manage PDFs" section, you can also:
     - **Search** for a PDF from the uploaded files.
     - **Rename** a PDF.
     - **Update** an existing PDF by uploading a new version.
     - **Delete** a PDF that you no longer need.

### WhatsApp Integration

1. **Run the WhatsApp Flask App**:
   - Start the WhatsApp integration by running the following command:
     ```bash
     python whatsapp.py
     ```
   - This will start the Flask server for WhatsApp on port `5001`.

2. **Expose the Server to the Internet**:
   - Use Ngrok or a similar tool to expose your local server to the internet:
     ```bash
     ngrok http 5001
     ```
   - Ngrok will provide you with a public URL.

3. **Set Up Twilio Webhook**:
   - In your Twilio console, configure the webhook URL to point to the Ngrok public URL followed by `/whatsapp`.
   - Example: `https://your-ngrok-url.ngrok.io/whatsapp`

4. **Interact via WhatsApp**:
   - Send a message to your Twilio WhatsApp number.
   - The chatbot will respond based on the content of the PDFs you’ve uploaded.
