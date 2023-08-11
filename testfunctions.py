# Testfunctions: Functions used for testing the generativeQA systems

def print_setup_details(llm_model, temp, embed_model, selected_prompt, index):
    openai.Engine.list()  # validate API access
    print(f"\nOpenAI API access validated!")
    print(f"LLM Model: {llm_model}\n"
          f"Temperature: {temp}\n"
          f"Embedding Model: {embed_model}\n"
          f"Selected Prompt: {selected_prompt}\n"
          f"Index Stats: {index.describe_index_stats()}")
    
print(f"Loaded {len(loader)} documents!\n")
    
def check_source_metadata_for_object(obj, method_name, query, item_type='item'):  # Checks if object has a 'source' metadata attribute
    items = getattr(obj, method_name)(query)  # Call the method to get items
    for item in items:
        source = None
        if hasattr(item, 'metadata'):  # Checks if item has metadata
            source = item.metadata.get('source')
        elif isinstance(item, dict):  # Checks if item is a dict
            source = item.get('source')
        if not source:  # If no source tag, print error
            id_str = ''
            if 'id' in item:
                id_str = f" with id {item['id']}"
            print(f"{item_type.capitalize()}{id_str} has an empty or non-existent 'source' value.")\
            
question = "Who is H.E.R.?"  # Question to retrieve most relevant docs

print(vectorstoredb.max_marginal_relevance_search(question, k=3))  # Tests MMR search and retrieval of docs

print(f"RetrievalQAChain Result: {qa.run(question)}\n")  # Tests RetrievalQAChain

print(f"Index Stats: {index.describe_index_stats()}")  # Print index stats
