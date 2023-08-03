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
'''


##### IMPORT NECESSARY MODULES FOR LANGCHAIN, PINECONE #####

import os
import sys
import time
import openai
import pinecone
import tiktoken
import traceback

from uuid import uuid4
from tqdm.auto import tqdm
from langchain.chains import RetrievalQA
from typing_extensions import Concatenate
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


##### OPENAI ACCESS + MODEL VARIABLES #####

llm_model = "gpt-4"  # OpenAI model
temp = 0.0  # Model temperature

print(f"LLM Model: {llm_model}, Model Temperature: {temp}", "\n") 

OPENAI_API_KEY = os.getenv('API_KEY') or 'API_KEY'
openai.api_key = OPENAI_API_KEY

print("OpenAI API access validated! \n\n\n")


##### BUILDING KNOWLEDGEBASE AND SETUP TOKENIZER #####

print("Building knowledgebase...\n")

# Load documents using DirectoryLoader
directory = 'data'

def load_docs(directory):
  loader = DirectoryLoader(directory, show_progress=True)
  res = loader.load()
  return res

docs = load_docs(directory)
print("\n", f"Docs: {docs}", "\n")

# Processing documents using tokenizer
tokenizer_name = tiktoken.encoding_for_model(llm_model)
tokenizer = tiktoken.get_encoding(tokenizer_name.name)

def tiktoken_len(text):  # Tiktoken_len returns length of tokens
    tokens = tokenizer.encode(
        text,
        disallowed_special = ()
    )
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 400,
    chunk_overlap = 20,
    length_function = tiktoken_len,
    separators = ["\n\n", "\n", " ", ""]
)

chunks = []
counter = 0  # Initialize a counter

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

embed = OpenAIEmbeddings(
    model = embed_model,
    openai_api_key = OPENAI_API_KEY
)

texts = [
    'this is the first chunk of text',
    'then another second chunk of text is here'
]

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
    pinecone.create_index(
        name = index_name,
        metric = 'cosine',
        dimension = len(res[0])  # 1536 dim of text-embedding-ada-002
    )

index = pinecone.Index(index_name)  # Connect to Pinecone index

index_stats = index.describe_index_stats()  # Get index stats
index_stats_str = str(index_stats)
print("Index Stats:\n\n", 
      '\n'.join('\t' + line for line in index_stats_str.split('\n')), "\n")  # View index stats

print("Index initialized! \n\n\n")


##### UPSERTING OUR DATA TO PINECONE INDEX #####

print("Upserting data to index... \n")

batch_size = 100  # how many embeddings we create and insert at once

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


# ##### RETRIEVAL #####

# print("Creating vector store and retrieving relevant sources...")

# text_field = "text"

# index = pinecone.Index(index_name)  # switch back to normal index for langchain

# vectorstore = Pinecone(index, embed.embed_query, text_field)

# primer = f"""You are a Q&A bot that translates English to French. 
#              If the information can not be found in the information
#              provided by the user you truthfully say "I don't know"."""

# llm = ChatOpenAI(model_name = llm_model,  
#                  openai_api_key = OPENAI_API_KEY,
#                  temperature = temp)

# qa = RetrievalQA.from_chain_type(  # LLM must answer  question based on vectorstore
#     llm = llm,
#     chain_type = "stuff",
#     retriever = vectorstore.as_retriever())
 
# def get_query():
#     query = input("Prompt: ")
#     return query

# def get_answer(query):  # Function to get answer
#     res = primer + " " + query # Combine the primer with the query
#     answer = "Result: " + qa.run(res) + "\n\n\n"
#     return answer

# def main():
#     while True:  # Main loop
#         query = get_query()
#         if query in ['quit', 'q', 'exit']:
#             sys.exit()
#         try:  # Get the answer using the question-answering chain
#             answer = get_answer(query)
#             print(answer)
#         except Exception as e:
#             print("An error occurred:", str(e))

# # system message to 'prime' the model
# primer = f"""You are Q&A bot. A highly intelligent system that answers
# user questions based on the information provided by the user above
# each question. If the information can not be found in the information
# provided by the user you truthfully say "I don't know".
# """

# res = openai.ChatCompletion.create(
#     model="gpt-4",
#     messages=[
#         {"role": "system", "content": primer},
#         {"role": "user", "content": augmented_query}
#     ]
# )

# If we drop the I don't know truthfully part from our primer, then we see something even worse than "I don't know" — hallucinations. 
# Clearly augmenting our queries with additional context can make a huge difference to the performance of our system.

index.delete(index_name)