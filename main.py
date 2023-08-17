# RETRIEVAL QA CONVERSATION CHAT AGENT - MAIN SCRIPT


# 1. Load

directory = 'DataDirectory'  # Directory to load data from

print(f"\n\n\nLoad:\nData directory selected: {directory}, Loading files...\n\n")

from Load.load_text import load_text  # Directory loader for text files

files = load_text(directory)  # Loads data from directory


# 2. Split & Store

print("\n\n\n2. Store:\n\n")

# 2.1 create pinecone index

from Setup.pinecone_variables import env, index_name, pine_key
from Store.init_pinecone_index import create_index, initialize_pinecone

initialize_pinecone(pine_key, env)  # Initializes Pinecone

index = create_index(index_name, 1536, 'cosine')  # Creates index

# 2.2 split and upsert files to pinecone index

from Setup.openai_variables import embed
from Store.upsert_to_pinecone import upsert_data_to_index

upsert_data_to_index(files, embed, index)  # Upserts splits to Pinecone index

# 2.3 Create VectorDB

from Store.create_pineconedb import create_pinecone_vectordb  # Pinecone DB

pinecone_vectordb = create_pinecone_vectordb(index_name, embed.embed_query)  # Initializes pinecone vectorstoredb


# 3. Retrieval

print("\n\n\n3. Retrieval:\n\n")

from Retrieval.pineconeretrieval import pineconeretrieval

query = "H.E.R."  # Test query to search for relevant docs

pineconeretriever = pineconeretrieval(pinecone_vectordb, query, "mmr")  # Initialize pinecone retriever with db and number of returned docs (k


# 4. Generate

# print("\n\n\n4. Generate:\n\n")

from Generate.retrievalqa_chain import create_retrievalqa_chain  # Retrieval QA Chain
from HelperFunctions.generatehelper import configure_retrievalqa_chain
from HelperFunctions.mainhelper import get_query
from Setup.openai_variables import llm

retrievalqa_chain = configure_retrievalqa_chain(create_retrievalqa_chain, llm, pineconeretriever)  # Retriever user's preferred chain type from input


# 5.  Main Loop for Testing GQA Quality of LoadQA and RetrievalQA 

from ConversationalAgent.prompts import get_prompt
from HelperFunctions.agenthelper import get_user_prompt


selected_prompt = get_prompt(get_user_prompt())  # User chooses prompt from prompts.py

system_message = selected_prompt  # Get system message from prompt

# 6. Create Agent

from ConversationalAgent.agent_setup import agent, conversational_memory, create_tools
from ConversationalAgent.conversation_agent import initialize_conversational_agent

conversational_agent = initialize_conversational_agent(agent, create_tools(retrievalqa_chain), llm, 
                                                    conversational_memory, 
                                                    system_message, True)  # Create conversational agent


# MAIN CHATBOT LOOP

from HelperFunctions.mainhelper import get_query

if __name__ == "__main__":  # Main prompt loop
    while True:
        query = get_query()
        if query.lower() in ['quit', 'q', 'exit']:
            break
        print("Vectorstore Chunks: ", pineconeretrieval(pinecone_vectordb, query, "mmr"),"\n\n\n")
        answer = "Agent Result: " + conversational_agent.run(query)
        if answer:
            print(answer)

index.delete(index_name)  # Delete index