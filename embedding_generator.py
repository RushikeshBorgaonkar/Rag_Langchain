from langchain_community.embeddings import HuggingFaceEmbeddings

def generate_embedding():
    
    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en')
    return embeddings



