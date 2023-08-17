def configure_retrievalqa_chain(retrievalqa, llm, retriever):
    while True:
        print("\nChoose a chain type for RetrievalQA:")
        print("1: Pine + RetrievalQA: Chaintype stuff")
        print("2: Pine + RetrievalQA: Chaintype map_reduce")
        print("3: Pine + RetrievalQA: Chaintype refine")
        print("4: Pine + RetrievalQA: Chaintype map_rerank")
            
        type_choice = input("Enter your choice (1-5): ")
        
        if type_choice == "1":
            return retrievalqa(llm, 'stuff', retriever)
        elif type_choice == "2":
            return retrievalqa(llm, 'map_reduce', retriever)
        elif type_choice == "3":
            return retrievalqa(llm, 'refine', retriever)
        elif type_choice == "4":
            return retrievalqa(llm, 'map_rerank', retriever)
        else:
            print("Invalid choice. Please select between 1-5.")