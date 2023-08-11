# **RetrievalQA Bot: Chat With Your Data**
### Powered by RetrievalQA-GPT4 + MMR Search to query local data using Pinecone & Langchain.


## Overview

This project is a demonstration of how to build a Conversational Agent powered by RetrievalQA-GPT4 + MMR Search to query directory files that are embedded and stored in a Vectorstore using Pinecone, Langchain, OpenAIEmbeddings, and Windows.

**QA over Documents**: This repository is modeled heavily after the QA over Documents [Documentation from Langchainin](https://python.langchain.com/docs/use_cases/question_answering/).

**OpenAI**: Provides AI models and tools for powering our chatbot, we will use GPT-4.

**Pinecone**: Pinecone is a vector database optimized for storing and querying vector embeddings, tailored for applications using large language models and generative AI.

**LangChain**: Provides modules for the management and optimization of language models in applications.

If you have text documents (like PDFs, blogs, Notion pages) and need to query their contents, this framework is perfect for the task.


## Getting Started

### Prerequisites

> :information_source: **Note** You must have access to GPT4 as well as a Pinecone account and API key to run this project. If you do not have access to GPT4, you must replace the GPT4 model in the project with another model that is available to you.

**I. WSL2 + VSCode**: [Windows Subsystem for Linux Installation Guide](https://docs.microsoft.com/en-us/windows/wsl/install-win10).
Ensure you have WSL2 installed and configured if using a Windows machine. I also recommend following along using it alongside VSCode as your text editor for this project.

**II. OpenAI Setup**: [OpenAI Quickstart Guide](https://platform.openai.com/docs/quickstart)

**Create an OpenAI API Key**: OpenAI API uses API keys for authentication. Ensure you have an OpenAI account and head to the API Keys page to create or retrieve the API key you'll use in your requests. -[OpenAI API Key](https://platform.openai.com/account/api-keys)

Once you have your API key, you can set it as an environment variable in an ubuntu wsl terminal:
```bash
export API_KEY="your_openai_api_key" # replace with your OpenAI API key
```

**III. Pinecone Setup**: [Pinecone Quickstart Guide](https://www.pinecone.io/quickstart/)

Create a Pinecone API Key: To use Pinecone, you must have an API key. To find your API key, open the Pinecone console and click API Keys. -[Pinecone Console](https://app.pinecone.io/)

Again, lets create another environment variable in your terminal for your Pinecone API key:
```bash
export PINE_KEY="your_pinecone_api_key" # replace with your Pinecone API key
```


### Installations
To install the required packages, run the following command:
```bash
pip3 install -U \ openai==0.27.7 \ pinecone-client \ pinecone-datasets==0.5.1 \ langchain==0.0.162 \ tiktoken==0.4.0 \ unstructured==0.7.12
```

### Understanding the Project Structure

We will frame our project around the following structure from [QA over Documents](https://python.langchain.com/docs/use_cases/question_answering/).
:

    Loading: Load data from sources as LangChain Document object.
    Splitting: Segment Documents into defined sizes.
    Storage: House and embed these splits, often in a vectorstore.
    Retrieval: Fetch splits from storage based on similarity to the query.
    Generation: Use an LLM to derive an answer combining the query and retrieved data.
    Conversation (Advanced): Enhance with Memory for multi-turn interactions.

### ***Heavily Inspired By:***
    Acknowledgment goes to Simon Cullen and TechleadHD for their leading examples. Much of the foundation and direction for this project were shaped by their contributions, as well as the comprehensive documentation from Langchain and Pinecone.

### Useful Links & References
    QA over Documents: Langchain Guide
    GPT-4 + Langchain Retrieval Augmentation: View on Github
    Pinecone Setup: Quickstart Guide
    Langchain Retrieval Augmentation: View on Colab
    Pinecone and Langchain Integration: Read the Documentation
    Accessing File Directory: Langchain Guide
    Text Embedding Models: Langchain Documentation
    Retrieval Agent: View on Github
    MMR Search: Research Paper
    RetrievalQA: Langchain Documentation