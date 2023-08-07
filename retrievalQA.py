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


# I. MODULES FOR LANGCHAIN/PINECONE

import os
import sys
import time
import openai
import pinecone

from langchain.agents import Tool, initialize_agent
from langchain.agents.types import AgentType
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from tqdm.auto import tqdm


# II. INITIAL SETUP

# a. OpenAI Access

OPENAI_API_KEY = os.getenv('API_KEY') or 'API_KEY'
openai.api_key = OPENAI_API_KEY
openai.Engine.list()

print(f"\nOpenAI API access validated!")

# b. OpenAI Variables

llm_model = "gpt-4"  # OpenAI LLM engine model
temp = 0.0  # Model temperature

print(f"LLM Model: {llm_model}, Model Temperature: {temp}\n\n\n")

embed_model = 'text-embedding-ada-002'  # Embedding model version
embed = OpenAIEmbeddings(model = embed_model, openai_api_key = OPENAI_API_KEY)  # Create embedding model

# c. Pinecone Variables

pine_key = os.environ.get('PINE_KEY')  # Pinecone key
env = "asia-southeast1-gcp-free"  # Pinecone environment


# 1. LOAD DOCUMENTS USING DIRECTORYLOADER

print("Building knowledgebase...\n")

directory = 'data'  # Directory of files to load

def load_data(directory):  # Load data from directory
  print("Loading documents:")
  loader = DirectoryLoader(directory, show_progress=True)
  res = loader.load()
  return res

files = load_data(directory)  # Directory files

for file in files:  # Check if original file has metadata
    if not file.metadata.get('source'):
        print(f"File with metadata {file.metadata} has an empty or non-existent 'source' value.")

print (f"Number of File Sources: {len(files)}\n")


# 2. SPLIT AND TAG DOCUMENT OBJECTS INTO TEXT SPLITS USING RECURSIVE TEXT SPLITTER

def split_docs(documents, chunk_size = 1000, chunk_overlap = 20):  # Split docs into text splits
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    splits = []  

    for document in documents:  # Iterate through each original document
        source = document.metadata.get('source', "DEFAULT_SOURCE")
        if source == "DEFAULT_SOURCE":
            print(f"Document {document} did not have a source. Setting default source.")
        source = os.path.splitext(os.path.basename(document.metadata['source']))[0]  # Format file source name
        splits_for_document = text_splitter.split_documents([document])  # Split this specific document
        splits.extend([{  
            'id': document.metadata['source'] + f'-{i}',  # Using 'source' as an ID
            'text': split.page_content,  # Page content from the split
            'source': source,  # Using 'source' as the URL
            'chunk': i  # Chunk
        } for i, split in enumerate(splits_for_document)])

    seen = set()  # Remove redundant splits based on text content
    new_splits = []  
    for split in splits:  
        text = split['text']  
        if text not in seen:  # If text is not in seen, add to new splits
            new_splits.append(split)
            seen.add(text)
    return new_splits

splits = split_docs(files)

for split in splits:  # Check if all splits have metadata
    if not split.get('source'):
        print(f"Split with id {split['id']} has an empty or non-existent 'source' value.")

print(f"Number of text splits: {len(splits)}\n")

print("Knowledgebase created!\n\n\n")


# 3. VECTOR DATABASE SETUP

# a. Test Embedding Model

print("Creating embedding model and vector store...\n")

texts = ['this is the first chunk of text',
         'then another second chunk of text is here']

res = embed.embed_documents(texts)  # Embed sample text documents

print(len(res), len(res[0]), "\n")  # Should return 2, 1536

# b. Create Pinecone Index

print("Initializing Index... \n")

pinecone.init(api_key = pine_key, environment = env)  # Initialize Pinecone

index_name = 'conversational-agent-retrievalqa'

if index_name not in pinecone.list_indexes():  # Create a new index
    pinecone.create_index(name = index_name,
                          metric = 'cosine',
                          dimension = len(res[0]))  # 1536 dim of text-embedding-ada-002

# c. Connect to Pinecone Index

index = pinecone.Index(index_name)  # Connect to Pinecone index

index_stats = index.describe_index_stats()  # Get index stats
index_stats_str = str(index_stats)  # Convert to string
print("Index Stats:\n\n", '\n'.join('\t' + line for line in index_stats_str.split('\n')), "\n")  # View index stats

print("Index initialized! \n\n\n")

# d. Upsert Vector Embeddings to Pinecone Index (Indexing)

print("Upserting data to index... \n")

batch_size = 100  # how many embeddings we create and insert at once

print("Populating index:")

# for i in tqdm(range(0, len(splits), batch_size)):  # find end of batch
#     i_end = min(len(splits), i + batch_size)
#     meta_batch = splits[i:i_end]  # get ids
#     ids_batch = [x['id'] for x in meta_batch]  # get texts to encode
#     texts = [x['text'] for x in meta_batch]  # create embeddings (try-except added to avoid RateLimitError)
#     try:  # try-except added to avoid RateLimitError
#         res = openai.Embedding.create(input = texts, engine = embed_model)  
#     except:
#         print(f"Exception occurred at batch starting at index {i}: {e}")
#         print(traceback.format_exc())
#         done = False
#         while not done:
#             time.sleep(5)
#             try:
#                 res = openai.Embedding.create(input = texts, engine = embed_model)
#                 done = True
#             except Exception as e:
#                 print(f"Rate limit hit at batch starting at index {i}: {e}")
#                 print(traceback.format_exc())
#     embeds = [record['embedding'] for record in res['data']]  # cleanup metadata
#     meta_batch = [{
#         'text': x['text'],
#         'chunk': x['chunk'],
#         'source': x['source']
#     } for x in meta_batch]
#     to_upsert = list(zip(ids_batch, embeds, meta_batch))  # upsert to Pinecone
#     index.upsert(vectors = to_upsert)

def create_embeddings(texts, embed_model):  # Define a helper function to create embeddings
    try:
        return openai.Embedding.create(input=texts, engine=embed_model)
    except Exception as e:
        print(f"Rate limit or other exception hit during embedding: {e}")
        time.sleep(5)  # Sleep for a while before retrying
        return create_embeddings(texts, embed_model)

for i in tqdm(range(0, len(splits), batch_size)):  # Main loop for upserting data to the index
    i_end = min(len(splits), i + batch_size)
    meta_batch = splits[i:i_end]
    ids_batch = [x['id'] for x in meta_batch]  # Extract ids and texts
    texts = [x['text'] for x in meta_batch]
    
    res = create_embeddings(texts, embed_model)  # Create embeddings
    embeds = [record['embedding'] for record in res['data']]
    
    meta_batch = [{  # Prepare metadata for upserting
        'text': x['text'],
        'chunk': x['chunk'],
        'source': x['source']
    } for x in meta_batch]
    
    for item in meta_batch:  # Check if any metadata in the batch lacks 'source' key
        if 'source' not in item:
            print(f"Item with id {item['id']} lacks 'source' key in metadata.")
    
    to_upsert = list(zip(ids_batch, embeds, meta_batch))  # Upsert to Pinecone
    try:
        index.upsert(vectors = to_upsert)
    except Exception as e:
        print(f"Error during Pinecone upsert at batch starting at index {i}: {e}")
        print("Problematic batch:", meta_batch)

print(f"\n Index populated! \n\n\n")


# 4. CREATING VECTORSTORE AND TESTING SEARCH

text_field = "text"

index = pinecone.Index(index_name)  # Switch back to normal index for langchain

vectorstoredb = Pinecone(index, embed.embed_query, text_field)  #Create vectorstore to use as index

vectorstoredb.similarity_search(" ", k = 3)  # Test to ensure a return of most relevant docs


# 4. RETRIEVAL

question = "Who is H.E.R.?"  # Question to retrieve most relevant docs

retriever = vectorstoredb.as_retriever(search_type = "mmr")  # Retrieve relevant splits using MMR search on vectorstore

retrieved_docs = retriever.get_relevant_documents(question, k = 3)
for doc in retrieved_docs:
    if not hasattr(doc, 'metadata') or 'source' not in doc.metadata:
        print(f"Document missing 'source' metadata: {doc}")


# 5. GENERATE RETRIEVED DOCS INTO ANSWERS USING LLM WITH RETRIEVALQA CHAIN

llm = ChatOpenAI(openai_api_key = OPENAI_API_KEY, model_name = llm_model,
                 temperature = temp)


retrievalqachain = RetrievalQAWithSourcesChain.from_chain_type(llm = llm, chain_type = "stuff",  # Question answering chain the LLM to VectorDB
                                 retriever = retriever, return_source_documents = True)  # LLM must answer the question based on VectorDB

print(retrievalqachain(question)) # Test to ensure a return of most relevant docs


# 6. CREATE CONVERSATIONAL AGENT AND CHAIN MEMORY

# a. Create conversational memory and tools

conversational_memory = ConversationBufferWindowMemory(memory_key = 'chat_history', 
                                                       k = 5, return_messages = True)

retrievalqa_desc = "Use this tool to answer questions that are relevant to local information and sources provided by the user"

tools = [Tool(name = 'RetrievalQA', func = retrievalqachain.run,
              description = retrievalqa_desc)]

# system_message = """

#                     You are a Q&A chat agent.

#                     You should always attach an important disclaimer before and after your response that you will be likely
#                     to generate misleading information and advice when assisting with requests outside of the local
#                     knowledge base provided.
                    
#                     You should only use the RetrievalQA tool to augment your responses to user questions relevant
#                     to locally stored information and sources.
                    
#                     If a user provides their own sources and asks for your advice or assistance, you
#                     should attempt to assist them with but add a disclamer that the accuracy of your response is at
#                     the user's discretion and quality of their sources.

#                     You should offer high quality assistance to the best of your ability to provide a good user experience
#                     but always provide disclaimers before responding to subjects and requests outside of your knowledgebase
#                     that you will potentially generate false or inaccurate responses as you must infer this knowledge
#                     using the LLM.
                    
#                     End every message with an iconic pop culture reference.
                     
#                 """

# b. Set boundaries and purpose of chat agent using system message

system_message = """

                    You are a Q&A chat agent.

                    You should only use the RetrievalQA tool to augment your responses to user questions relevant
                    to locally stored information and sources.

                    You should offer high quality assistance to the best of your ability to provide a good user experience.

                 """  # Change system prompt to modify or restrict behavior of chat agent

# c. Initialize conversational agent

conversational_agent = initialize_agent(agent = AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                                        tools = tools,
                                        llm = llm,
                                        memory = conversational_memory,
                                        agent_kwargs = {"system_message": system_message},
                                        verbose = True,
                                        return_intermediate_steps = True)

print("Conversational agent initialized! \n\n\n")


# III. CHATBOT LOOP

def get_query():
    query = input("Ask a question! (type 'quit', 'q', or 'exit' to quit): ")
    if len(query) > 1000:  # Prevent excessively long input
        print("Your input is too long. Please try again.")
        return get_query()
    else:
        return str(query)

def get_agentchain_answer(query):  # Function to get conversational agent answer
    try:  # Get answer using the qa agent chain
        answer = "Result: " + conversational_agent.run(query)
        return answer
    except Exception as e:
        print("An error occurred while getting the answer:", str(e), "\n")
        return None

def main():  # Main loop
    while True:
        query = get_query()
        if query in ['quit', 'q', 'exit']:
            sys.exit()
        answer = get_agentchain_answer(query)
        if answer is not None and answer.strip() != '':
            print(answer, "\n")

main()

print("Chatbot Ended.")

index.delete(index_name)