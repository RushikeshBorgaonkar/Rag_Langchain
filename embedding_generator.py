from langchain_community.embeddings import HuggingFaceEmbeddings

class EmbeddingGenerator:
    def __init__(self, model_name='BAAI/bge-small-en'):
        self.model = HuggingFaceEmbeddings(model_name=model_name)

    def generate_embedding(self, text):
        return self.model.embed_documents([text])[0]
