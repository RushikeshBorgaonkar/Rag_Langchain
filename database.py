import os
from dotenv import load_dotenv
import psycopg2
import numpy as np
from langchain.memory import ConversationBufferMemory
from collections import deque

load_dotenv()

class ChatHistoryManager:
    def __init__(self, max_size=5):
        self.history = []  # List to store chat history
        self.max_size = max_size  # Maximum size of the history

    def add_chat(self, user_message, assistant_message):
        """Add a chat entry to the history."""
        self.history.append((user_message, assistant_message))
        if len(self.history) > self.max_size:
            self.history.pop(0)  # Remove the oldest entry if exceeding max size

    def get_last_chats(self):
        """Retrieve the last chats."""
        return self.history

class DatabaseManager:
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASS'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT', '5432')
        )
        self.create_tables()
        self.chat_history_manager = ChatHistoryManager(max_size=5)  # Initialize the custom chat history manager

    def create_tables(self):
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    text_id VARCHAR(255),
                    text_content TEXT,
                    embedding vector(384)
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    query TEXT,
                    response TEXT
                );
            """)
            self.conn.commit()

    def add_embedding_to_db(self, embedding, text_id, text_content):
        with self.conn.cursor() as cur:
            embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            cur.execute(
                "INSERT INTO embeddings (text_id, text_content, embedding) VALUES (%s, %s, %s)",
                (text_id, text_content, embedding_list)
            )
            self.conn.commit()

    def search_similar_vectors(self, query_embedding, top_k=2):
        with self.conn.cursor() as cur:
            query_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
            cur.execute("""
                SELECT text_content, 1 - (embedding <=> %s::vector) as similarity
                FROM embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (query_list, query_list, top_k))
            
            results = cur.fetchall()
            return [(text, float(score)) for text, score in results]

    def add_chat_to_db(self, query, response):
        """Add a chat entry to the database and to the custom history manager."""
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_history (query, response) VALUES (%s, %s)",
                (query, response)
            )
            self.conn.commit()
        
        # Store the chat in the custom history manager
        self.chat_history_manager.add_chat(query, response)

    def get_last_five_chats(self):
        """Retrieve the last 5 chats."""
        return self.chat_history_manager.get_last_chats()

    def get_chat_history(self):
        """Retrieve all chat history from the database."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT query, response FROM chat_history ORDER BY id DESC;")
            db_history = cur.fetchall()
        
        return db_history  # Return only database history

    def close(self):
        self.conn.close()

    def clear_embeddings(self):
        """Clear all existing embeddings from the database"""
        with self.conn.cursor() as cursor:
            cursor.execute("TRUNCATE TABLE embeddings")
        self.conn.commit()
