import pinecone
from langchain.vectorstores import Pinecone

def initialize_vectorstore(index_name, embed_function):
    index = pinecone.Index(index_name)  # Switch back to normal index for langchain
    vectorstore = Pinecone(index, embed_function, "text")  # Creates vectorstoredb
    return vectorstore