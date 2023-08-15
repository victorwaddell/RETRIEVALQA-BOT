def pineconeretrieval(vectordb, query, search_type):
    retriever = vectordb.as_retriever(search_type = search_type)
    print(f"Retrieved Pinecone docs: {retriever.get_relevant_documents(query)}\n")  # Test retriever
    return retriever