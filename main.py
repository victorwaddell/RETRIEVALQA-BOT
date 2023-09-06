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
import gradio as gr

title = """
<div style="text-align: center;max-width: 700px;">
    <h1>Chat with PDF</h1>
    <p style="text-align: center;">Upload a .PDF from your computer, click the "Load PDF to LangChain" button, <br />
    when everything is ready, you can start asking questions about the pdf ;)</p>
    <a style="display:inline-block; margin-left: 1em" href="https://huggingface.co/spaces/fffiloni/langchain-chat-with-pdf?duplicate=true"><img src="https://img.shields.io/badge/-Duplicate%20Space%20to%20skip%20the%20queue-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14" alt="Duplicate Space"></a>
</div>
"""

def bot(query):
    print("Debug: Entering bot function")

    if not query:
        print("Debug: query is None or empty!")
        return "Please enter a valid query."

    print(f"Debug: Starting retrieval for query: {query}")
    retrieval_result = pineconeretrieval(pinecone_vectordb, query, "mmr")
    print("Debug: Retrieval completed")

    print("Debug: Starting conversational agent")
    agent_result = conversational_agent.run(query)
    print("Debug: Conversational agent completed")

    return f"Vectorstore Chunks: {retrieval_result}\nAgent Result: {agent_result}"



def update_chatbot(query=None):
    print("Debug: update_chatbot triggered")  # Add this line
    global chat_history  # Use the global chat_history list
    if query is None:
        print("Debug: query is None")
    else:
        print(f"Debug: query is {query}")

    if query:
        print("Debug: About to call bot function")
        chat_response = bot(query)
        print("Debug: Bot function returned")
        print(f"Debug: Chat response is {chat_response}")

        # Your chatbot update logic
        print("Debug: About to update chatbot")
        chatbot.set_message([
            (entry["message"], entry["role"]) for entry in chat_history])
        print("Debug: Chatbot updated")

# Gradio Interface
with gr.Blocks() as app:
    chatbot = gr.Chatbot([], height=350)
    question = gr.Textbox(label="Question", placeholder="Type your question and hit Enter")
    submit_btn = gr.Button("Send message")

    # Link the submit and click actions to the update_chatbot function
    question.submit(update_chatbot)
    submit_btn.click(update_chatbot)

app.launch()

index.delete(index_name)  # Delete index