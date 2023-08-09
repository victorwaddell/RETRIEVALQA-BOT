from langchain.document_loaders import DirectoryLoader

def load_data(directory):  # Loads data from directory
    print("Loading documents:")
    loader = DirectoryLoader(directory, show_progress = True)
    return loader.load()  # Returns list of documents