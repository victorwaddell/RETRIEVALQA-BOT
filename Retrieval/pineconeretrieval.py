def pineconeretrieval(vectordb, query, search_type):
    retriever = vectordb.as_retriever(search_type = search_type)
    return retriever