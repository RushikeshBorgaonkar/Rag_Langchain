import os
from flask import Flask, request, render_template, session
from database import initialize_chat_history, add_chat_to_db,get_retriever, get_recent_chat_history, add_embedding_to_db # Import functions from database.py

from embedding_generator import generate_embedding
from langchain_core.runnables import RunnablePassthrough

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

def load_text_samples(file_paths):
    texts = []
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for file_path in file_paths:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            for doc in documents:
              
                chunks = text_splitter.split_text(doc.page_content)
                texts.extend(chunks)

        elif file_path.startswith("https://www.youtube.com/watch?v="):
            loader = YoutubeLoader.from_youtube_url(file_path)
            documents = loader.load()
            for doc in documents:
                chunks = text_splitter.split_text(doc.page_content)
                texts.extend(chunks)

        elif file_path.endswith(".mp3") or file_path.endswith(".wav"):
            loader = AssemblyAIAudioTranscriptLoader(file_path)
            documents = loader.load()
            for doc in documents:     
                chunks = text_splitter.split_text(doc.page_content)
                texts.extend(chunks)

    return texts

def process_embeddings(texts, source=None):
    for idx, text in enumerate(texts):
        text_id = f"{source}_{idx}" if source else str(idx)
        add_embedding_to_db(text_id, text) 

def generate_augmented_response(query):
    llm = ChatGroq(
        model="mixtral-8x7b-32768",  
        temperature=0.5,
        max_tokens=1024,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant that provides comprehensive answers by analyzing and synthesizing information from multiple documents.Context:\n{context}",
            ),
            ("human", """Answer the {question} using the context"""),
        ]
    )
    retriever = get_retriever()
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm    
    )
    result = rag_chain.invoke(query)

    # retriever = get_retriever()
   
    # similarity_search=retriever.invoke(query)
    # print(f"Retrieved doc : {similarity_search}")
    # rag_chain =  prompt| llm
        
    # result = rag_chain.invoke({"context": similarity_search, "question": query})
    generated_response = result.content
    return {
        "query": query,
        "generated_response": generated_response
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    session_id = session.get('session_id', None)
    chat_history = initialize_chat_history(session_id)  
    session['session_id'] = chat_history._session_id  

    if request.method == 'POST':
        
        query_input = QueryInput(query=request.form['query'])
        print(query_input)
        
       
        texts = load_text_samples(DATA_FILE_PATHS)
          
        # process_embeddings(texts)
  
        result = generate_augmented_response(query_input.query)
        
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