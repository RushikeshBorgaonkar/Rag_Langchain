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

def generate_augmented_response(query: str, retrieved_items: List[tuple[str, float]]):
    # Initialize the text splitter
    text_splitter = CharacterTextSplitter(
        chunk_size=512,  # Set the desired chunk size
        chunk_overlap=50  # Set the desired overlap between chunks
    )

    # Chunk the retrieved documents
    all_chunks = []
    for idx, (text, _) in enumerate(retrieved_items):
        chunks = text_splitter.split_text(text)  # Use Langchain's text splitter
        all_chunks.extend(chunks)  # Add the chunks to the list

    # Create context from the chunks
    context = "\n\n".join(f"Document {idx + 1}:\n{chunk}" for idx, chunk in enumerate(all_chunks))
    
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
            ("human", f"Please provide a comprehensive response to the question using information from ALL the provided documents below.\n\nContext Documents:\n{context}\n\nQuestion: {query}\nPlease provide a detailed response incorporating information from all documents."),
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

            db_manager.clear_embeddings()

            texts = load_text_samples(DATA_FILE_PATHS)
            processed_texts = set()
            for idx, text in enumerate(texts):
                if text in processed_texts:
                    continue
                processed_texts.add(text)
                
                embedding = embedding_gen.generate_embedding(text)
                db_manager.add_embedding_to_db(embedding, text_id=str(idx), text_content=text)
                print(f"Added embedding for: {text}")  # Print added embedding to terminal

            query_embedding = embedding_gen.generate_embedding(query_text)
            similar_items = db_manager.search_similar_vectors(query_embedding, top_k=2)

            result = generate_augmented_response(query_text, similar_items)
            
            # Print the result to the terminal
            print(f"\nQuery Result:\nQuery: {result['query']}")
            print(f"\nGenerated Response:\n{result['generated_response']}")

            # Store the result in the session
            session['result'] = result
            
            # Store chat history in the database
            db_manager.add_chat_to_db(query_text, result['generated_response'])

            # Retrieve chat history from the database
            chat_history = db_manager.get_chat_history()

            # Render the result in the browser
            return render_template_string('''<h1>Query Result</h1>
                <h2>Query: {{ result.query }}</h2>
                <h3>Generated Response:</h3>
                <p>{{ result.generated_response }}</p>
                <h3>Chat History:</h3>
                <ul>
                    {% for chat in chat_history %}
                        <li><strong>Query:</strong> {{ chat[0] }} <br> <strong>Response:</strong> {{ chat[1] }}</li>
                    {% endfor %}
                </ul>
                <form method="POST">
                    <input type="text" name="query" placeholder="Enter your next query" required>
                    <button type="submit">Submit</button>
                </form>
                <a href="/">Back</a>
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