import os
from flask import Flask, request, render_template, session
from database import initialize_chat_history, add_chat_to_db, get_recent_chat_history, add_embedding_to_db  # Import functions from database.py
from embedding_generator import generate_embedding
from langchain_groq import ChatGroq
from pytube import YouTube  
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import AssemblyAIAudioTranscriptLoader
from langchain_community.document_loaders import YoutubeLoader
from langchain.document_loaders import PyPDFLoader
from typing import List
from langchain.text_splitter import CharacterTextSplitter
from models import QueryInput, AugmentedResponse  # Import Pydantic models from models.py

app = Flask(__name__)
app.secret_key = os.urandom(24)  

DATA_FILE_PATHS = [
    'C:/Users/Coditas-Admin/Desktop/ALL FOLDERS/VINOD GIVEN ASSIGNMENTS/RAG USING LANGCHAIN/DATA/virus-warning-scam-25323.mp3',
    'https://www.youtube.com/watch?v=cNGjD0VG4R8',
    'C:/Users/Coditas-Admin/Desktop/ALL FOLDERS/VINOD GIVEN ASSIGNMENTS/RAG USING LANGCHAIN/DATA/attention_all_you_need.pdf'
   
]

def load_text_samples(file_paths: List[str]) -> List[str]:
    texts = []
    for file_path in file_paths:
        
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            for doc in documents:
                texts.append(doc.page_content)
            print(f"Documents for PDF : {documents}")

    
        elif file_path.startswith("https://www.youtube.com/watch?v="): 
            print(f"Processing YouTube video from {file_path}")
            loader = YoutubeLoader.from_youtube_url(file_path)
            documents = loader.load()
            for doc in documents:
                texts.append(doc.page_content)
                print(f"APPENDED TEXT FOR VIDEO : {texts[-1]}")
            print(f"Documents for video : {documents}")

   
        elif file_path.endswith(".mp3") or file_path.endswith(".wav"):  
            print(f"Processing audio file from {file_path}")
            loader = AssemblyAIAudioTranscriptLoader(file_path)
            documents = loader.load()
            for doc in documents:
                texts.append(doc.page_content)
                print(f"APPENDED TEXT FOR AUDIO : {texts[-1]}")
            print(f"Documents for AUDIO : {documents}")
    
    return texts

def process_embeddings(texts):
    for idx, text in enumerate(texts):
        text_id = str(idx)    
        add_embedding_to_db(text_id, text)  

def generate_augmented_response(query: str, retrieved_items: List[tuple[str,str]], last_five_context: str):
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,  
        chunk_overlap=100
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
        
        query_input = QueryInput(query=request.form['query'])
        
       
        texts = load_text_samples(DATA_FILE_PATHS)
        print(f"Extracted texts: {texts}")
          
        process_embeddings(texts)
        
        last_five_chats = get_recent_chat_history(chat_history)
        print(f"Last five chats: {last_five_chats}")
        last_five_context = "\n\n".join(f"User: {chat[0]}" for chat in last_five_chats)
        print(f"Last five context: {last_five_context}")

        retrieved_items = last_five_chats 

        result = generate_augmented_response(query_input.query, retrieved_items, last_five_context)

        structured_result = AugmentedResponse(**result)  
        session['result'] = structured_result.dict()  
        add_chat_to_db(chat_history, query_input.query, structured_result.generated_response) 
        print(f"Generated result: {structured_result}")
        chat_history = get_recent_chat_history(chat_history)  

        return render_template('chatbot.html', result=structured_result.dict(), chat_history=chat_history)  

    return '''
            <form method="POST">
                <input type="text" name="query" placeholder="Enter your query" required>
                <button type="submit">Submit</button>
            </form>
        '''

if __name__ == "__main__":
    app.run(debug=True)
