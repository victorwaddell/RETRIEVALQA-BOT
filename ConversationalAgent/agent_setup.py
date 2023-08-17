from langchain.agents import Tool
from langchain.agents.types import AgentType
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from Generate.retrievalqa_chain import retrievalqa_desc  # Imports tool description of QA system

conversational_memory = ConversationBufferWindowMemory(memory_key = 'chat_history',  # Creates chain memory 
                                                       k = 5, return_messages = True)

agent = AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION

def create_tools(retrievalqa_chain):  # Creates tools for conversational agent
    return [Tool(name = 'RetrievalQA', func = retrievalqa_chain.run, 
                 description = retrievalqa_desc)]

