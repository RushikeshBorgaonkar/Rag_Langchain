
import os
import uuid
from dotenv import load_dotenv
import psycopg
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector as VectorStore
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from embedding_generator import generate_embedding  
from langchain_postgres import PostgresChatMessageHistory

load_dotenv()

# Initialize global variables
DB_HOST = os.environ.get("DB_HOST")
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASS = os.environ.get("DB_PASS")
DB_PORT = os.environ.get("DB_PORT", "5432")  


connection = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

collection_name = "embeddings"


def initialize_chat_history(session_id=None):
    table_name = "chat_history"
    
    
    sync_connection = psycopg.connect(connection)
    
    PostgresChatMessageHistory.create_tables(sync_connection, table_name)
    
    if session_id is None:
        session_id = str(uuid.uuid4())
    
    chat_history = PostgresChatMessageHistory(
        table_name,
        session_id,
        sync_connection=sync_connection
    )
    
    return chat_history

def get_recent_chat_history(chat_history):
    """Retrieve the last 5 messages from chat history as tuples of (query, response)."""
    messages = chat_history.messages[-12:] if len(chat_history.messages) > 12 else chat_history.messages  # Get last 10 messages
    chat_pairs = []

   
    for i in range(0, len(messages), 2): 
        if i + 1 < len(messages):  
            user_message = messages[i]
            ai_message = messages[i + 1]
            if isinstance(user_message, HumanMessage) and isinstance(ai_message, AIMessage):
                chat_pairs.append((user_message.content, ai_message.content))

    return chat_pairs[-5:]  

def add_message_to_history(chat_history, role, content):
   
    if role == "system":
        message = SystemMessage(content=content)
    elif role == "ai":
        message = AIMessage(content=content)
    elif role == "human":
        message = HumanMessage(content=content)
    else:
        raise ValueError("Invalid role specified")
    
    chat_history.add_message(message)

def add_embedding_to_db(text_id, text_content):
    """Add an embedding to the vector store."""
    embedding = generate_embedding()  # Assuming this generates the embeddings
    vector_store = VectorStore(
        embeddings=embedding,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )
    metadata = {"text_id": text_id}
    vector_store.add_texts([text_content], metadatas=[metadata]) 


def add_chat_to_db(chat_history, query, response):
    """Add a chat entry to the database."""
    add_message_to_history(chat_history, "human", query)  
    add_message_to_history(chat_history, "ai", response)   
    

def get_retriever():

    embedding_model = generate_embedding() 
    

    vector_store = VectorStore(
        embeddings=embedding_model,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )
    
    results = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    return results