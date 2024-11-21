import os
from flask import Flask, request, render_template, session
from database import initialize_chat_history, add_chat_to_db, get_recent_chat_history, add_embedding_to_db  # Import functions from database.py
from embedding_generator import generate_embedding
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from typing import List
from langchain.text_splitter import CharacterTextSplitter


app = Flask(__name__)
app.secret_key = os.urandom(24)  

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

def process_embeddings(texts):
   
    for idx, text in enumerate(texts):
        text_id = str(idx)    
        add_embedding_to_db(text_id, text)  
        
       
def generate_augmented_response(query: str, retrieved_items: List[tuple[str,str]], last_five_context: str):
    text_splitter = CharacterTextSplitter(
        chunk_size=2000,  
        chunk_overlap=200  
    )

    all_chunks = []
    for idx, (text, second_text) in enumerate(retrieved_items):  
        chunks = text_splitter.split_text(text)  
        all_chunks.extend(chunks)   

        for chunk in chunks:
            text_id = f"{idx}_{chunks.index(chunk)}"
            add_embedding_to_db(text_id, chunk)  

        second_chunks = text_splitter.split_text(second_text)  
        for second_chunk in second_chunks:
            second_text_id = f"{idx}_second_{second_chunks.index(second_chunk)}"
            add_embedding_to_db(second_text_id, second_chunk)  

    context = f"{last_five_context}\n\n" + "\n\n".join(f"Document {idx + 1}:\n{chunk}" for idx, chunk in enumerate(all_chunks))
    print(context)
    llm = ChatGroq(
        model="mixtral-8x7b-32768",  
        temperature=0.5,
        max_tokens=1024,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant that provides comprehensive answers by analyzing and synthesizing information from multiple documents.",
            ),
            ("human", f"Using the following context, please provide a comprehensive response to the question:\n\nContext:\n{context}\n\nQuestion: {query}"),
        ]
    )

    chain = prompt | llm
    ai_msg = chain.invoke({"input": prompt}) 

    response = ai_msg.content.strip()  
    return {
        "query": query,
        "generated_response": response
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    session_id = session.get('session_id', None)
    chat_history = initialize_chat_history(session_id)  
    session['session_id'] = chat_history._session_id  

    if request.method == 'POST':
        query_text = request.form['query']

        texts = load_text_samples(DATA_FILE_PATHS)
        process_embeddings(texts)
        
        last_five_chats = get_recent_chat_history(chat_history)
        print(f"last five chats are : {last_five_chats}")
        last_five_context = "\n\n".join(f"User: {chat[0]}" for chat in last_five_chats)
        print(f"last_five_context are :{last_five_context} ")

       
        retrieved_items = last_five_chats  

        result = generate_augmented_response(query_text, retrieved_items, last_five_context)

        session['result'] = result
        
        add_chat_to_db(chat_history, query_text, result['generated_response'])  

        chat_history = get_recent_chat_history(chat_history) 

        return render_template('chatbot.html', result=result, chat_history=chat_history)  

    return '''
            <form method="POST">
                <input type="text" name="query" placeholder="Enter your query" required>
                <button type="submit">Submit</button>
            </form>
        '''
   

if __name__ == "__main__":
    app.run(debug=True)
