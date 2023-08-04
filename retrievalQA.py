# RetrievalQA example

'''

This script sets up the environment for using the OpenAI GPT-4 model, Pinecone vector search engine, and CMU's MMR search to perform 
document similarity search.

Pinecone enables developers to build scalable, real-time recommendation and search systems based on vector similarity search. 

LangChain, on the other hand, provides modules for managing and optimizing the use of language models in applications.

First, it loads and processes a set of documents, then embeds the documents, and stores the embedded documents in a Pinecone index.
Following this, an MMR search is ran on the docs for efficiency and relevancy to the user query.

Credits goto Simon Cullen for his work and many Langchain docs are referenced.

Refer to the following for more information:
- gpt4 + langchain retrieval augmentation: https://github.com/pinecone-io/examples/blob/master/generation/gpt4-retrieval-augmentation/gpt-4-langchain-docs.ipynb
- Pinecone setup: https://docs.pinecone.io/docs/quickstart
- Langchain retrieval augmentation: https://colab.research.google.com/github/pinecone-io/examples/blob/master/generation/langchain/handbook/05-langchain-retrieval-augmentation.ipynb#scrollTo=0RqIF2mIDwFu
- Pinecone Langchain integration: https://python.langchain.com/docs/integrations/vectorstores/pinecone
- Accessing file directory: https://python.langchain.com/docs/modules/data_connection/document_loaders/file_directory
- Text embedding models: https://python.langchain.com/docs/modules/data_connection/text_embedding/
- Retrieval agent: https://github.com/pinecone-io/examples/blob/master/generation/langchain/handbook/08-langchain-retrieval-agent.ipynb
- MMR Search: https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf
'''


##### IMPORT NECESSARY MODULES FOR LANGCHAIN, PINECONE #####

import os
import sys
import time
import traceback

import openai
import pinecone
import tiktoken
from langchain.agents import Tool, initialize_agent
from langchain.agents.types import AgentType
from langchain.chains import RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from tqdm.auto import tqdm


##### OPENAI ACCESS + MODEL VARIABLES #####

llm_model = "gpt-4"  # OpenAI model
temp = 0.0  # Model temperature

print(f"LLM Model: {llm_model}, Model Temperature: {temp}", "\n") 

OPENAI_API_KEY = os.getenv('API_KEY') or 'API_KEY'
openai.api_key = OPENAI_API_KEY

print("OpenAI API access validated!\n\n\n")


##### BUILDING KNOWLEDGEBASE AND SETUP TOKENIZER #####

print("Building knowledgebase...\n")

directory = 'data'  # Load documents using DirectoryLoader

def load_docs(directory):
  print("Loading documents:")
  loader = DirectoryLoader(directory, show_progress=True)
  res = loader.load()
  return res

docs = load_docs(directory)

# Process documents using tokenizer
tokenizer_name = tiktoken.encoding_for_model(llm_model)
tokenizer = tiktoken.get_encoding(tokenizer_name.name)

def tiktoken_len(text):  # Tiktoken_len returns length of tokens
    tokens = tokenizer.encode(text, disallowed_special = ())
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 400, chunk_overlap = 20,
    length_function = tiktoken_len,
    separators = ["\n\n", "\n", " ", ""])

chunks = []

print("\n","Chunking documents:\t")
for document in tqdm(docs):  # Process the docs into more chunks
    if len(document.page_content) < 200:  # If page content is short we can skip
        continue
    texts = text_splitter.split_text(document.page_content)
    source = os.path.splitext(os.path.basename(document.metadata['source']))[0]  # Format file source name nicely
    chunks.extend([{'id': document.metadata['source'] + f'-{i}',  # Assuming 'source' can be used as an ID
                    'text': texts[i],
                    'url': source,  # Assuming 'source' is the URL
                    'chunk': i} 
                    for i in range(len(texts))])

print("\n", f"Length of chunks: {len(chunks)}", "\n")

print("Knowledgebase created! \n\n\n")


##### CREATE EMBEDDINGS #####

print("Creating embedding model \n")

embed_model = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(model = embed_model, openai_api_key = OPENAI_API_KEY)

texts = ['this is the first chunk of text',
         'then another second chunk of text is here']

res = embed.embed_documents(texts)

print(len(res), len(res[0]), "\n")

print("Embedding model initialized! \n\n\n")


##### INITIALIZING THE INDEX #####

print("Initializing Index... \n")

pine_key = os.environ.get('PINE_KEY')  # Pinecone key
env = "asia-southeast1-gcp-free"  # Pinecone environment

pinecone.init(api_key = pine_key, environment = env)

index_name = 'gpt-4-langchain-docs'

if index_name not in pinecone.list_indexes():  # Create a new index
    pinecone.create_index(name = index_name,
                          metric = 'cosine',
                          dimension = len(res[0]))  # 1536 dim of text-embedding-ada-002

index = pinecone.Index(index_name)  # Connect to Pinecone index

index_stats = index.describe_index_stats()  # Get index stats
index_stats_str = str(index_stats)
print("Index Stats:\n\n", 
      '\n'.join('\t' + line for line in index_stats_str.split('\n')), "\n")  # View index stats

print("Index initialized! \n\n\n")


##### UPSERTING OUR DATA TO PINECONE INDEX #####

print("Upserting data to index... \n")

batch_size = 100  # how many embeddings we create and insert at once

print("Populating index:")

for i in tqdm(range(0, len(chunks), batch_size)):  # find end of batch
    i_end = min(len(chunks), i + batch_size)
    meta_batch = chunks[i:i_end]  # get ids
    ids_batch = [x['id'] for x in meta_batch]  # get texts to encode
    texts = [x['text'] for x in meta_batch]  # create embeddings (try-except added to avoid RateLimitError)
    try:
        res = openai.Embedding.create(input = texts, engine = embed_model)
    except:
        print(f"Exception occurred at batch starting at index {i}: {e}")
        print(traceback.format_exc())
        done = False
        while not done:
            time.sleep(5)
            try:
                res = openai.Embedding.create(input = texts, engine = embed_model)
                done = True
            except Exception as e:
                print(f"Rate limit hit at batch starting at index {i}: {e}")
                print(traceback.format_exc())
    embeds = [record['embedding'] for record in res['data']]  # cleanup metadata
    meta_batch = [{
        'text': x['text'],
        'chunk': x['chunk'],
        'url': x['url']
    } for x in meta_batch]
    to_upsert = list(zip(ids_batch, embeds, meta_batch))  # upsert to Pinecone
    index.upsert(vectors = to_upsert)

print("\n", "Index populated! \n\n\n")


##### CREATING VECTOR STORE AND QUERYING #####

text_field = "text"

index = pinecone.Index(index_name)  # Switch back to normal index for langchain

vectorstoredb = Pinecone(index, embed.embed_query, text_field) #create vectorstore to use as index

query = "Who is H.E.R."

print(vectorstoredb.similarity_search(query, k = 3), "\n")  # Test to ensure a return of 3 most relevant docs


##### INITIALIZE RETRIEVALQA OBJECT (GQA) #####

llm = ChatOpenAI(openai_api_key = OPENAI_API_KEY, model_name = llm_model,
                 temperature = temp)

qa = RetrievalQA.from_chain_type(llm = llm, chain_type = "stuff",  # Question answering chain the LLM to VectorDB
                                 retriever = vectorstoredb.as_retriever())  # LLM must answer the question based on VectorDB


##### CREATING CONVERSATIONAL AGENT #####

conversational_memory = ConversationBufferWindowMemory(memory_key = 'chat_history', 
                                                       k = 5, return_messages = True)

retrievalqa_desc = "Use this tool to answer questions that are relevant to local information and sources provided by the user"

tools = [Tool(name = 'RetrievalQA', func = qa.run,
              description = retrievalqa_desc)]

system_message = """You are a Q&A chat agent.

                    You should always attach an important disclaimer before and after your response that you will be likely
                    to generate misleading information and advice when assisting with requests outside of the local
                    knowledge base provided.
                    
                    You should only use the RetrievalQA tool to answer user questions that are directly relevant
                    to locally stored information provided. If relevant information cannot be found using the RetrievalQA
                    tool, alert the user that you do not have access to the information locally and provide a disclaimer
                    that you will generate a response using the LLM which could lead to misinformation or inaccuracy.
                    
                    If a user provides their own sources and asks directly for your advice or assistance, you
                    should attempt to assist them with but add a disclamer that the accuracy of your response is at
                    the user's discretion and quality of sources provided.

                    You should offer high quality assistance to the best of your ability to provide a good user experience
                    but always provide disclaimers before responding to subjects and requests outside of your knowledgebase
                    that you will potentially generate false advice or inaccurate answers as you must infer this
                    knowledge using the LLM.
                    
                    End every message with an iconic pop culture reference.
                     
                """

conversational_agent = initialize_agent(agent = AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                                        tools = tools,
                                        llm = llm,
                                        memory = conversational_memory,
                                        agent_kwargs = {"system_message": system_message},
                                        verbose = True)

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