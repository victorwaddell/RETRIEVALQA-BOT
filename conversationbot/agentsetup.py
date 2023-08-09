from langchain.agents import Tool
from langchain.agents.types import AgentType
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from generate.retrievalqachain import retrievalqa_desc  # Imports tool description of QA system

conversational_memory = ConversationBufferWindowMemory(memory_key = 'chat_history',  # Creates chain memory 
                                                       k = 5, return_messages = True)

agent = AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION

def create_tools(qa_instance):  # Creates tools for conversational agent
    return [Tool(name = 'RetrievalQA', func = qa_instance.run, 
                 description = retrievalqa_desc)]

