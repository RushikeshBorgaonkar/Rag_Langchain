from langchain_community.embeddings import HuggingFaceEmbeddings

def generate_embedding():
    """Generate an embedding model instance."""
    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en')
    return embeddings



