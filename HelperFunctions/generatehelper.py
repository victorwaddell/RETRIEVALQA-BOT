def configure_retrievalqa_chain(retrievalqa, llm, retriever):
    while True:
        print("\nChoose a chain type for RetrievalQA:")
        print("1: RetrievalQA Type stuff")
        print("2: RetrievalQA Type map_reduce")
        print("3: RetrievalQA Type refine")
        print("4: RetrievalQA Type map_rerank")
        print("5: Go back")
            
        type_choice = input("Enter your choice (1-5): ")
        
        if type_choice == "1":
            return ("RetrievalQA - stuff", retrievalqa(llm, 'stuff', retriever))
        elif type_choice == "2":
            return ("RetrievalQA - map_reduce", retrievalqa(llm, 'map_reduce', retriever))
        elif type_choice == "3":
            return ("RetrievalQA - refine", retrievalqa(llm, 'refine', retriever))
        elif type_choice == "4":
            return ("RetrievalQA - map_rerank", retrievalqa(llm, 'map_rerank', retriever))
        elif type_choice == "5":
            return (None, None)
        else:
            print("Invalid choice. Please select between 1-5.")