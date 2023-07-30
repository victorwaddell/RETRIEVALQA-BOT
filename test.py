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


##### Import necessary modules for OpenAI, PineCone, GPT4 #####

import os
import time
import openai
import pinecone
import tiktoken

from uuid import uuid4
from tqdm.auto import tqdm
from datasets import load_dataset # Example dataset
from typing_extensions import Concatenate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from IPython.display import display, Markdown
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


##### OpenAI Access + Model Variables #####

print("Ensuring OpenAI access...")

openai_key = os.environ.get('API_KEY')  # OpenAI key 

# openai.api_key = openai_key  # Set openai key directly in library
# openai.Engine.list()  # Testing API access

llm_model = "gpt-4"  # OpenAI model
temp = 0.0  # Model temperature

print("OpenAI API access validated! \n\n\n")

##### Building Knowledgebase and Setup Tokenizer #####

print("Building knowledgebase...")

data = load_dataset("wikipedia", "20220301.simple", split='train[:10000]')
data

print("\t dataset loaded")

tokenizer_name = tiktoken.encoding_for_model(llm_model)
tokenizer = tiktoken.get_encoding(tokenizer_name.name)

def tiktoken_len(text):  # Tiktoken_len returns length of tokens
    tokens = tokenizer.encode(
        text,
        disallowed_special = ()
    )
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400, chunk_overlap=20,
    length_function = tiktoken_len,
    separators = ["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_text(data[6]['text'])[:3]

print(chunks)

print(tiktoken_len(chunks[0]), tiktoken_len(chunks[1]), tiktoken_len(chunks[2]))

print("Knowledgebase created! \n\n\n")


##### Creating Embeddings #####

print("Creating embeddings...")

model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model = model_name,
    openai_api_key = openai_key
)

texts = [
    'this is the first chunk of text',
    'then another second chunk of text is here'
]
 
res = embed.embed_documents(texts)

print(len(res), len(res[0]))

print("Embedding creation complete! \n\n\n")


##### Vector Database Setup #####

print("Creating Pinecone vector DB...")

pine_key = os.environ.get('PINE_KEY')  # Pinecone key
env = "asia-southeast1-gcp-free"  # Pinecone environment

pinecone.init(api_key=pine_key, environment=env)

index_name = 'gpt-4-langchain-docs'

if index_name not in pinecone.list_indexes():
    pinecone.create_index(  # Create a new index
        name = index_name,
        metric = 'cosine',
        dimension = len(res[0])  # 1536 dim of text-embedding-ada-002
    )

index = pinecone.Index(index_name)  # Connect to index
print(index.describe_index_stats())  # View index stats

print("Vector DB created! \n\n\n")


##### Populate Pinecone Index with our Langchain Docs #####

print("Performing indexing...")

batch_limit = 100

texts = []
metadatas = []

for i, record in enumerate(tqdm(data)):
    metadata = {  # First get metadata fields for this record
        'wiki-id': str(record['id']), 
        'source': record['url'],
        'title': record['title']
    }
    record_texts = text_splitter.split_text(record['text'])  # Now we create chunks from the record text
    record_metadatas = [{  # Create individual metadata dicts for each chunk
        "chunk": j, "text": text, **metadata
    } for j, text in enumerate(record_texts)]  # Append these to current batches
    texts.extend(record_texts)
    metadatas.extend(record_metadatas)
    if len(texts) >= batch_limit:  # If we have reached the batch_limit we can add texts
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embed.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds, metadatas))
        texts = []
        metadatas = []

if len(texts) > 0:
    ids = [str(uuid4()) for _ in range(len(texts))]
    embeds = embed.embed_documents(texts)
    index.upsert(vectors=zip(ids, embeds, metadatas))

print(index.describe_index_stats())

print("Pinecone index populated! \n\n\n")


##### Creating a Vector Store and Querying #####

print("Creating vector store and retrieving relevant sources...")

text_field = "text"

index = pinecone.Index(index_name)  # Switch back to normal index for langchain

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

query = "how do I use the LLMChain in LangChain?"

vectorstore.similarity_search(  # Pinecone + Langchain
    query,  # our search query
    k=3  # return 3 most relevant docs
)

print("Retrieval complete! \n\n\n")


##### RetrievalQA example #####

llm = ChatOpenAI(model_name = llm_model,  
                 openai_api_key = openai_key,
                 temperature = temp)  

qa = RetrievalQA.from_chain_type(  # LLM must answer  question based on vectorstore
    llm = llm,
    chain_type = "stuff",
    retriever = vectorstore.as_retriever()
)

print(f"Pinecone docs + Langchain Result:\n{qa.run(query)} \n\n\n")

# directory = 'chatgpt-retrieval/data'
# loader = DirectoryLoader(directory, use_multithreading = True)  
# documents = loader.load()