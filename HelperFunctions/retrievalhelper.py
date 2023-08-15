

def display_search_results(search_function, retriever, query, k = 3):
    print(f"--- {retriever} ---")
    results = search_function(query, k = k)
    for result in results:
        print(result)
    print("\n")
    
def user_selection(vectorstore_db, docs, query):
    print("Please select a search method to test the retrievers (Exit to proceed to testing chains):")
    print("1: VectorStore DB MMR Search")
    print("2: VectorStore DB Similarity Search")
    print("3: Docs MMR Search")
    print("4: Docs Similarity Search")
    print("5: Exit")
    
    choice = input("Enter your choice (1-5): ")
    
    if choice == "1":
        display_search_results(vectorstore_db.max_marginal_relevance_search, "VectorStore DB MMR Search", query)
    elif choice == "2":
        display_search_results(vectorstore_db.similarity_search, "VectorStore DB Similarity Search", query)
    elif choice == "3":
        display_search_results(docs.max_marginal_relevance_search, "Docs MMR Search", query)
    elif choice == "4":
        display_search_results(docs.similarity_search, "Docs Similarity Search", query)
    elif choice == "5":
        print("Exiting...")
        return
    else:
        print("Invalid choice. Please select between 1-5.")

    user_selection(vectorstore_db, docs, query)  # Recursively call with passed arguments.