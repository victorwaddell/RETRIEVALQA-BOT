def chromaretrieval(vectordb, query):
    retriever = vectordb.as_retriever()
    print(f"Retrieved Chroma docs: {retriever.get_relevant_documents(query)}\n")  # Test retriever
    return retriever