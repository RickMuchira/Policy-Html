# Policy-Html

This project is a web-based application that allows users to interact with a chatbot for asking questions related to uploaded PDFs. Additionally, there is WhatsApp integration that provides the same functionality via a WhatsApp interface.

## Features

- **PDF Management**: Upload, manage, and update PDFs.
- **Chatbot Interface**: Ask questions related to the uploaded PDFs and receive relevant answers.
- **WhatsApp Integration**: Interact with the chatbot via WhatsApp.
- **Real-time Processing**: Uses Google and GROQ APIs for processing and responding to queries.

## Project Structure

.
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
