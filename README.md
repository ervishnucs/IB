<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Insurance Chatbot</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f9f9f9;
        }
        h1 {
            color: #007bff;
            text-align: center;
        }
        h2 {
            color: #333;
        }
        p {
            color: #555;
            line-height: 1.6;
        }
        pre {
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }
        ul {
            list-style-type: disc;
            margin-left: 20px;
            color: #333;
        }
        .container {
            max-width: 800px;
            margin: auto;
            padding: 20px;
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>AI Insurance Chatbot</h1>
        <p>This project is an AI-driven chatbot designed to assist users with queries related to insurance policy documents. Built with FastAPI and LangChain, it leverages advanced natural language processing techniques to provide accurate and concise responses.</p>

        <h2>Features</h2>
        <ul>
            <li>Interactive chatbot interface for user inquiries.</li>
            <li>Capable of answering questions about insurance policies based on provided documents.</li>
            <li>Utilizes embeddings for effective document retrieval and understanding.</li>
            <li>Supports conversation history to improve user experience.</li>
        </ul>

        <h2>Requirements</h2>
        <ul>
            <li>Python 3.8+</li>
            <li>FastAPI</li>
            <li>Uvicorn</li>
            <li>LangChain</li>
            <li>Pydantic</li>
            <li>Hugging Face Transformers</li>
            <li>Additional dependencies for document loading and embeddings.</li>
        </ul>

        <h2>Installation</h2>
        <p>1. <strong>Clone the repository:</strong></p>
        <pre>
   git clone https://github.com/yourusername/ai-insurance-chatbot.git
   cd ai-insurance-chatbot
        </pre>
    </div>
</body>
</html>
