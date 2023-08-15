import pinecone

def initialize_pinecone(api_key, environment):  # Initializes Pinecone
    pinecone.init(api_key = api_key, environment = environment)
    
def create_index(index_name, dimension, metric):  # Creates Pinecone index
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name = index_name, metric = metric, 
                              dimension = dimension)
    index = pinecone.Index(index_name)
    print("Pinecone Index Created!")
    return index