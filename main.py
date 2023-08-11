# RETRIEVAL QA CONVERSATION CHAT AGENT - MAIN SCRIPT


# 1. Load

from load.loaddirectory import load_data

directory = 'data'  # Directory to load data from

files = load_data(directory)  # Loads data from directory


# 2. Split

from split.splitfiles import split_docs

splits = split_docs(files, 1500, 20)  # Split documents (files, chunk_size, chunk_overlap)


# 3. Store

from setup.openaivariables import embed_model
from setup.pineconevariables import env, index_name, pine_key
from store.init_index import create_index, initialize_pinecone
from store.upserttopinecone import upsert_data_to_index

# 3.1 Create Pinecone Index

initialize_pinecone(pine_key, env)  # Initializes Pinecone

index = create_index(index_name, 1536, 'cosine')  # Creates index

# 3.2 Upsert to Index

upsert_data_to_index(splits, 100, embed_model, index)  # Upserts data to index


# 4. Retrieval

from retrieval.vectordb import initialize_vectorstore
from setup.openaivariables import embed

vectorstore_db = initialize_vectorstore(index_name, embed.embed_query)  # Initializes vectorstore


# 5. Generation

from generate.retrievalqachain import create_qa_chain
from setup.openaivariables import llm
from setup.qa_variables import chain_type, search_type

qa = create_qa_chain(vectorstore_db, search_type, llm, chain_type)  # Create QA system (vectorstore, reranker, pine_key, llm_model, temp, prompt_type)


# 6. Conversational Agent

# 6.1 Choose Prompt

from conversationbot.prompts import get_prompt
from mainhelper import get_user_prompt

default_prompt = """You are a chatbot that answers questions about local information and sources provided by the user. Mention that you are the default chatbot choice at the start of conversation."""

selected_prompt = get_prompt(get_user_prompt())  # User chooses prompt from prompts.py

system_message = selected_prompt  # Get system message from prompt

# 6.2 Create Agent

from conversationbot.agentsetup import (agent, conversational_memory,
                                        create_tools)
from conversationbot.conversationagent import initialize_conversational_agent

conversational_agent = initialize_conversational_agent(agent, create_tools(qa), llm, 
                                                    conversational_memory, 
                                                    system_message, True)  # Create conversational agent


# MAIN CHATBOT LOOP

from mainhelper import get_query, get_response

if __name__ == "__main__":  # Main prompt loop
    while True:
        query = get_query()
        if query.lower() in ['quit', 'q', 'exit']:
            break
        print(f"Vectorstore Chunks: {vectorstore_db.similarity_search(query, k = 3)}\n\n\n")
        print(f"RetrievalQAChain Result: {qa.run(query)}\n\n\n")
        answer = "Result: " + conversational_agent.run(query)
        if answer:
            print(f"""MMR Search and Retrieval of Docs: {answer}\n\n\n""")
            
print("Chatbot Ended.")

index.delete(index_name)  # Delete index to save resources