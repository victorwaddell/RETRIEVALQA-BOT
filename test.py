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

openai.api_key = openai_key  # Set openai key directly in library
openai.Engine.list()  # Testing API access

llm_model = "gpt-4"  # OpenAI model
temp = 0.0  # Model temperature

print("OpenAI API access validated! \n\n\n")

##### Building Knowledgebase and Setup Tokenizer #####

print("Building knowledgebase...")

docs = load_dataset('jamescalam/langchain-docs-23-06-27', split='train')  # Example dataset testing
docs

print("\t dataset loaded")

chunks = []  # List to store chunks
tokenizer_name = tiktoken.encoding_for_model('gpt-4')
tokenizer = tiktoken.get_encoding(tokenizer_name.name)

def tiktoken_len(text):  # Tiktoken_len returns length of tokens
    tokens = tokenizer.encode(
        text,
        disallowed_special = ()
    )
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter( 
    chunk_size = 500,  chunk_overlap = 20,
    length_function = tiktoken_len,
    separators = ["\n\n", "\n", " ", ""]
)

for page in tqdm(docs):  # Process the docs into more chunks using this approach.
    if len(page['text']) < 200:  # If page content is short we can skip
        continue
    texts = text_splitter.split_text(page['text'])
    chunks.extend([{
        'id': page['id'] + f'-{i}',
        'text': texts[i],
        'url': page['url'],
        'chunk': i
    } for i in range(len(texts))])

print("\t processed docs into chunks")

print("Knowledgebase created! \n\n\n")


##### Creating Embeddings #####

print("Creating embeddings...")

embed_model = "text-embedding-ada-002"

embed = OpenAIEmbeddings(
    model = embed_model,
    openai_api_key = openai_key
)

texts = [
    'this is the first chunk of text',
    'then another second chunk of text is here'
]
 
res = embed.embed_documents(texts)

print("Embedding creation complete! \n\n\n")


##### Vector Database Setup #####

print("Creating Pinecone vector DB...")

pine_key = os.environ.get('PINE_KEY')  # Pinecone key
env = "asia-southeast1-gcp-free"  # Pinecone environment

pinecone.init(api_key=pine_key, environment=env)

index_name = 'gpt-4-langchain-docs'

print(len(res[0]))  # Testing if dimensions is correct

if index_name not in pinecone.list_indexes():  # Check if index already exists
    print("Please allow some time for index to create")
    pinecone.create_index(index_name, dimension = len(res[0]), metric = 'cosine')

index = pinecone.Index(index_name)  # Connect to index
index.describe_index_stats()  # View index stats

print("Vector DB created! \n\n\n")


##### Populate Pinecone Index with our Langchain Docs #####

print("Performing indexing...")

batch_size = 100  # How many embeddings we create and insert at once

for i in tqdm(range(0, len(chunks), batch_size)):
    i_end = min(len(chunks), i + batch_size)  # Find end of batch
    meta_batch = chunks[i:i_end]  # Get ids
    ids_batch = [x['id'] for x in meta_batch]  # Get texts to encode
    texts = [x['text'] for x in meta_batch]  # Create embeddings (try-except added to avoid RateLimitError)
    try:
        res = openai.Embedding.create(input = texts, engine = embed_model)
    except:
        done = False
        while not done:
            time.sleep(5)
            try:
                res = openai.Embedding.create(input = texts, engine = embed_model)
                done = True
            except:
                pass
    embeds = [record['embedding'] for record in res['data']]
    meta_batch = [{  # Cleanup metadata
        'text': x['text'],
        'chunk': x['chunk'],
        'url': x['url']
    } for x in meta_batch]
    to_upsert = list(zip(ids_batch, embeds, meta_batch))  # Upsert to Pinecone
    index.upsert(vectors=to_upsert)

print("Pinecone index populated! \n\n\n")


##### Creating a Vector Store and Querying #####

print("Creating vector store and retrieving relevant sources...")

query = "how do I use the LLMChain in LangChain?"

res = openai.Embedding.create(input = [query], engine = embed_model)  # Gpt4 + Langchain
xq = res['data'][0]['embedding']  # Retrieve from Pinecone
res = index.query(xq, top_k=5, include_metadata = True)  # Get relevant contexts (including the questions)

print("Retrieval complete! \n\n\n")


##### Augmentation and Answer Generation #####

print("Augmenting query...")

contexts = [item['metadata']['text'] for item in res['matches']]  # Get list of retrieved text
augmented_query = "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"+query

# OpenAI's Chat API example

primer = f"""You are Q&A bot. A highly intelligent system that answers
user questions based on the information provided by the user above
each question. If the information can not be found in the information
provided by the user you truthfully say "I don't know".
"""

res = openai.ChatCompletion.create(
    model = llm_model,
    messages=[  # System message to 'prime' the model
        {"role": "system", "content": primer},
        {"role": "user", "content": augmented_query}
    ]
)

print(f"Gpt-4 Langchain & Pinecone Result:\n {res} \n\n\n")

# directory = 'chatgpt-retrieval/data'
# loader = DirectoryLoader(directory, use_multithreading = True)  
# documents = loader.load()