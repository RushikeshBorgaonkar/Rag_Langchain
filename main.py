import os
from flask import Flask, request, render_template_string, session
from database import DatabaseManager
from embedding_generator import EmbeddingGenerator
from langchain_groq import ChatGroq  # Import ChatGroq from langchain_groq
from langchain_core.prompts import ChatPromptTemplate  # Import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from typing import List
from langchain.text_splitter import CharacterTextSplitter

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secret key for session management

DATA_FILE_PATHS = [
    'C:/Users/Coditas-Admin/Desktop/ALL FOLDERS/VINOD GIVEN ASSIGNMENTS/RAG USING LANGCHAIN/DATA/attention_all_you_need.pdf', 
    'C:/Users/Coditas-Admin/Desktop/ALL FOLDERS/VINOD GIVEN ASSIGNMENTS/RAG USING LANGCHAIN/DATA/PEPSI-2022-Presentation1.pdf',
    'C:/Users/Coditas-Admin/Desktop/ALL FOLDERS/VINOD GIVEN ASSIGNMENTS/RAG USING LANGCHAIN/DATA/q1-2022-pep_transcript.pdf'
]

def load_text_samples(file_paths):
    texts = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file '{file_path}' not found.")
        
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        for doc in documents:
            texts.append(doc.page_content)
            print(f"Extracted from {file_path}: {doc.page_content}")
    
    return list(dict.fromkeys(filter(None, texts)))

def process_embeddings(texts, db_manager):
    embedding_gen = EmbeddingGenerator()  # Create an instance of EmbeddingGenerator
    for idx, text in enumerate(texts):
        text_id = str(idx)  # Use a unique identifier for each text
        if not db_manager.embedding_exists(text_id):  # Check if the embedding already exists
            embedding = embedding_gen.generate_embedding(text)  # Generate the embedding
            db_manager.add_embedding_to_db(embedding, text_id=text_id, text_content=text)  # Store the embedding in the database
            print(f"Added embedding for: {text}")  # Print added embedding to terminal
        else:
            print(f"Embedding already exists for: {text_id}. Retrieving existing embedding.")

def generate_augmented_response(query: str, retrieved_items: List[tuple[str, float]], db_manager: DatabaseManager, embedding_gen: EmbeddingGenerator, last_five_context: str):
    # Initialize the text splitter
    text_splitter = CharacterTextSplitter(
        chunk_size=5000,  # Set the desired chunk size
        chunk_overlap=500  # Set the desired overlap between chunks
    )

    # Chunk the retrieved documents and store embeddings
    all_chunks = []
    for idx, (text, _) in enumerate(retrieved_items):  # Ensure retrieved_items has the correct structure
        chunks = text_splitter.split_text(text)  # Use Langchain's text splitter
        all_chunks.extend(chunks)  # Add the chunks to the list

        # Process each chunk to create embeddings
        for chunk in chunks:
            text_id = f"{idx}_{chunks.index(chunk)}"  # Create a unique ID for each chunk
            if not db_manager.embedding_exists(text_id):  # Check if the embedding already exists
                embedding = embedding_gen.generate_embedding(chunk)  # Use the embedding generator
                db_manager.add_embedding_to_db(embedding, text_id=text_id, text_content=chunk)
                print(f"Added embedding for chunk: {chunk}")  # Print added embedding to terminal
            else:
                print(f"Embedding already exists for chunk: {text_id}. Retrieving existing embedding.")

    # Create context from the chunks and last 5 chats
    context = f"{last_five_context}\n\n" + "\n\n".join(f"Document {idx + 1}:\n{chunk}" for idx, chunk in enumerate(all_chunks))
    
    # Initialize ChatGroq client with the API key
    llm = ChatGroq(
        model="mixtral-8x7b-32768",  # Specify your model
        temperature=0.5,
        max_tokens=1024,
    )

    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant that provides comprehensive answers by analyzing and synthesizing information from multiple documents.",
            ),
            ("human", f"Using the following context, please provide a comprehensive response to the question:\n\nContext:\n{context}\n\nQuestion: {query}"),
        ]
    )

    # Create a chain from the prompt and the LLM
    chain = prompt | llm

    # Invoke the chain with the necessary parameters
    ai_msg = chain.invoke({
        "input_language": "English",  # You can adjust this as needed
        "output_language": "English",  # You can adjust this as needed
        "input": query,  # Pass the query as input
    }) 

    # Extract the response
    response = ai_msg.content.strip()  # Adjusted to access the response correctly
    
    return {
        "query": query,
        "generated_response": response
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    db_manager = DatabaseManager()  # Initialize the database manager
    try:
        if request.method == 'POST':
            query_text = request.form['query']
            embedding_gen = EmbeddingGenerator()

            # Clear previous embeddings if needed
            db_manager.clear_embeddings()

            # Load text samples and process embeddings
            texts = load_text_samples(DATA_FILE_PATHS)
            process_embeddings(texts, db_manager)  # Call the process_embeddings function

            # Retrieve the last 5 chats
            last_five_chats = db_manager.get_last_five_chats()
            last_five_context = "\n\n".join(f"User: {chat[0]}\nAssistant: {chat[1]}" for chat in last_five_chats)

            # Example of how to populate retrieved_items correctly
            retrieved_items = []  # This should be populated based on your logic
            # Populate retrieved_items with tuples of (text, score)
            # For example:
            retrieved_items.append(("Some text", 0.85))  # Add your logic here

            # Generate a response using the last 5 chats as context
            result = generate_augmented_response(query_text, retrieved_items, db_manager, embedding_gen, last_five_context)

            # Store the result in the session
            session['result'] = result
            
            # Store chat history in the database
            db_manager.add_chat_to_db(query_text, result['generated_response'])

            # Retrieve chat history from the database
            chat_history = db_manager.get_last_five_chats()  # Get only the last 5 chats

            # Render the result in the browser
            return render_template_string('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chatbot</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 600px;
            margin: auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        h1 {
            text-align: center;
            color: #333;
        }
        h2 {
            color: #555;
        }
        h3 {
            color: #777;
        }
        .response {
            background-color: #e9ecef;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
        }
        .chat-history {
            margin: 20px 0;
            border-top: 1px solid #ccc;
            padding-top: 10px;
        }
        .chat-item {
            margin-bottom: 10px;
            padding: 10px;
            border-radius: 5px;
            background-color: #f8f9fa;
        }
        .chat-item strong {
            display: block;
            color: #333;
        }
        form {
            display: flex;
            justify-content: space-between;
            margin-top: 20px;
        }
        input[type="text"] {
            flex: 1;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
            margin-right: 10px;
        }
        button {
            padding: 10px 15px;
            border: none;
            border-radius: 5px;
            background-color: #007bff;
            color: white;
            cursor: pointer;
        }
        button:hover {
            background-color: #0056b3;
        }
        a {
            display: inline-block;
            margin-top: 20px;
            text-decoration: none;
            color: #007bff;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Chatbot</h1>
        <h2>Query: {{ result.query }}</h2>
        <h3>Generated Response:</h3>
        <p class="response">{{ result.generated_response }}</p>
        <div class="chat-history">
            <h3>Last 5 Chats:</h3>
            <ul>
                {% for chat in chat_history %}
                    <li class="chat-item">
                        <strong>Query:</strong> {{ chat[0] }} <br>
                        <strong>Response:</strong> {{ chat[1] }}
                    </li>
                {% endfor %}
            </ul>
        </div>
        <form method="POST">
            <input type="text" name="query" placeholder="Enter your next query" required>
            <button type="submit">Submit</button>
        </form>
        <a href="/">Back</a>
    </div>
</body>
</html>
''', result=result, chat_history=chat_history)

        return '''
            <form method="POST">
                <input type="text" name="query" placeholder="Enter your query" required>
                <button type="submit">Submit</button>
            </form>
        '''
    finally:
        db_manager.close()  # Ensure the connection is closed after all operations

if __name__ == "__main__":
    app.run(debug=True)