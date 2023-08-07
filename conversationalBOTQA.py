# RETRIEVAL QA CHAT AGENT

'''

Q&A Script to build Conversational Agent powered by RetrievalQA + MMR Search to query directory files that are embedded and stored in a Vectorstore
using Pinecone + Langchain. The script will be able to use the splits retrieved from the process to augment OpenAI GPT-4's answers.

Pinecone enables developers to build scalable, real-time recommendation and search systems based on vector similarity search. 

LangChain, on the other hand, provides modules for managing and optimizing the use of language models in applications.

First, it loads and processes a set of documents, then embeds the documents, and stores the embedded documents in a Pinecone index.
Following this, an MMR search is ran on the docs for efficiency and relevancy to the user query.

Credits goto Simon Cullen and TechleadHD for their work, as well as many Langchain and Pinecone docs.

Refer to the following for more information:
- gpt4 + langchain retrieval augmentation: https://github.com/pinecone-io/examples/blob/master/generation/gpt4-retrieval-augmentation/gpt-4-langchain-docs.ipynb
- Pinecone setup: https://docs.pinecone.io/docs/quickstart
- Langchain retrieval augmentation: https://colab.research.google.com/github/pinecone-io/examples/blob/master/generation/langchain/handbook/05-langchain-retrieval-augmentation.ipynb#scrollTo=0RqIF2mIDwFu
- Pinecone Langchain integration: https://python.langchain.com/docs/integrations/vectorstores/pinecone
- Accessing file directory: https://python.langchain.com/docs/modules/data_connection/document_loaders/file_directory
- Text embedding models: https://python.langchain.com/docs/modules/data_connection/text_embedding/
- Retrieval agent: https://github.com/pinecone-io/examples/blob/master/generation/langchain/handbook/08-langchain-retrieval-agent.ipynb
- MMR Search: https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf
- QA over Documents: https://python.langchain.com/docs/use_cases/question_answering/
- RetrievalQA: https://python.langchain.com/docs/modules/chains/retrievalqa/

'''


# I. IMPORTS

import os
import sys
import time
import openai
import pinecone
from tqdm.auto import tqdm
from langchain.agents import Tool, initialize_agent
from langchain.agents.types import AgentType
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone


# II. SETUP

# a. OpenAI Access

OPENAI_API_KEY = os.getenv('API_KEY') or 'API_KEY'
openai.api_key = OPENAI_API_KEY
openai.Engine.list()  # validate API access

print(f"\nOpenAI API access validated!")

# b. OpenAI Variables

llm_model = "gpt-4"  # OpenAI LLM engine model
temp = 0.0  # OpenAI LLM temperature

embed_model = 'text-embedding-ada-002'  # OpenAI Embedding model
embed = OpenAIEmbeddings(model = embed_model, openai_api_key = OPENAI_API_KEY)  # Embedding variable

pine_key = os.environ.get('PINE_KEY')  # Pinecone API key
env = "asia-southeast1-gcp-free"  # Pinecone environment

directory = 'data'  # Directory to load documents from
index_name = 'conversational-agent-retrievalqa'  # Pinecone index name


# III. HELPER FUNCTIONS


def load_data(directory):  # Loads data from directory
    print("Loading documents:")
    loader = DirectoryLoader(directory, show_progress = True)
    return loader.load()  # Returns list of documents


def split_docs(documents, chunk_size = 1000, chunk_overlap = 20):  # Splits documents into text splits
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    splits = []

    for document in documents:
        source = document.metadata.get('source', "DEFAULT_SOURCE")
        if source == "DEFAULT_SOURCE":
            print(f"Document {document} did not have a source. Setting default source.")
        source = os.path.splitext(os.path.basename(document.metadata['source']))[0]
        splits_for_document = text_splitter.split_documents([document])
        splits.extend([{  # Adds split + metadata to splits list
            'id': document.metadata['source'] + f'-{i}',
            'text': split.page_content,
            'source': source,
            'chunk': i
        } for i, split in enumerate(splits_for_document)])

    seen = set()  # Remove redundant splits based on text content
    new_splits = []  
    for split in splits:  
        text = split['text']  
        if text not in seen:  # If text is not in seen, add to new splits
            new_splits.append(split)
            seen.add(text)
    return new_splits


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
            print(f"{item_type.capitalize()}{id_str} has an empty or non-existent 'source' value.")


def create_embeddings(texts, embed_model):  # Creates embeddings for texts
    try:  # Tries to create embeddings
        return openai.Embedding.create(input = texts, engine = embed_model)
    except Exception as e:  # Prints error if embedding fails
        print(f"Rate limit or other exception hit during embedding: {e}")
        time.sleep(5)
        return create_embeddings(texts, embed_model)  # Retries embedding
    

def upsert_data_to_index(splits, batch_size, embed_model, index):  # Upserts data to Pinecone index
    for i in tqdm(range(0, len(splits), batch_size)):  # Iterates through splits
        i_end = min(len(splits), i + batch_size)  # Gets end of batch
        meta_batch = splits[i:i_end]
        ids_batch = [x['id'] for x in meta_batch]  # Gets ids, texts, and embeddings for batch
        texts = [x['text'] for x in meta_batch]
        res = create_embeddings(texts, embed_model)
        embeds = [record['embedding'] for record in res['data']]

        to_upsert = list(zip(ids_batch, embeds, meta_batch))  # Zips ids, embeddings, and metadata
        try:  # Tries to upsert data
            index.upsert(vectors = to_upsert)
        except Exception as e:  # Prints error if upsert fails
            print(f"Error during Pinecone upsert at batch starting at index {i}: {e}\nProblematic batch: {meta_batch}")


def get_query():  # Gets user query
    query = input("Ask a question! (type 'quit', 'q', or 'exit' to quit): ")
    return query if len(query) <= 1000 else get_query()  # Checks if query is too long


def get_agentchain_answer(query):  # Gets answer from agentchain
    try:
        return "Result: " + conversational_agent.run(query)  # Returns answer
    except Exception as e:  
        print(f"An error occurred while getting the answer: {str(e)}")
        return None


# IV. MAIN SCRIPT

# 1. Building Knowledgebase

files = load_data(directory)  # Loads data from directory

# check_source_metadata(files, item_type = 'file')  # Checks if metadata is valid for each file

# 2. Splitting Documents

splits = split_docs(files)  # Splits documents

# check_source_metadata(splits, item_type = 'split')

# 3. Vector Database

pinecone.init(api_key = pine_key, environment = env)  # Initializes Pinecone

if index_name not in pinecone.list_indexes():  # Creates index if it doesn't exist
    pinecone.create_index(name = index_name, metric = 'cosine', dimension = 1536)

index = pinecone.Index(index_name)  # Connect to index

upsert_data_to_index(splits, 100, embed_model, index)  # Upserts data to index

# 4. Vectorstore And Testing Search

index = pinecone.Index(index_name)  # Switch back to normal index for langchain

vectorstoredb = Pinecone(index, embed.embed_query, "text")  # Creates vectorstoredb

print(vectorstoredb.similarity_search(" ", k = 3))  # Tests search and vectorstore

# 5. Retrieval

question = "Who is H.E.R.?"  # Question to retrieve most relevant docs

retriever = vectorstoredb.as_retriever(search_type = "mmr")  # Creates retriever with MMR search

check_source_metadata_for_object(retriever, "get_relevant_documents", question, item_type = 'doc')  # Checks if metadata is valid for each doc

# 6. Generation

llm = ChatOpenAI(openai_api_key = OPENAI_API_KEY,  # Creates LLM  
                 model_name = llm_model, temperature = temp)  

qa = RetrievalQA.from_chain_type(llm = llm, chain_type = "stuff",  # Question answering chain the LLM to VectorDB
                                 retriever = vectorstoredb.as_retriever())  # LLM must answer the question based on VectorDB

print(f"RetrievalQAChain Result: {qa.run(question)}\n")

# 7. Creating Conversational Agent

conversational_memory = ConversationBufferWindowMemory(memory_key = 'chat_history',  # Creates chain memory 
                                                       k = 5, return_messages = True)

retrievalqa_desc = "Use this tool to answer questions that are relevant to local information and sources provided by the user"

tools = [Tool(name = 'RetrievalQA', func = qa.run,  # Creates RetrievalQA tool 
              description = retrievalqa_desc)]

system_message = """
    You are a Q&A chat agent.
    You should only use the RetrievalQA tool to augment your responses to user questions relevant to locally stored information and sources.
    You should offer high quality assistance to the best of your ability to provide a good user experience.
"""  # System message to prompt chat agent and define its boundaries and behavior

conversational_agent = initialize_agent(  # Creates conversational agent
    agent = AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    tools = tools,
    llm = llm,
    memory = conversational_memory,
    agent_kwargs = {"system_message": system_message},
    verbose = True)


# V. CHATBOT LOOP

if __name__ == "__main__":  # Main prompt loop
    while True:
        query = get_query()
        if query.lower() in ['quit', 'q', 'exit']:
            break
        answer = get_agentchain_answer(query)
        if answer:
            print(answer, "\n")

print("Chatbot Ended.")

index.delete(index_name)