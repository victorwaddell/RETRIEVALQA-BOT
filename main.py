# RETRIEVAL QA CONVERSATION CHAT AGENT - MAIN SCRIPT


# 1. Load

directory = 'DataDirectory'  # Directory to load data from

print(f"\n\n\nLoad:\nData directory selected: {directory}, Loading files...\n\n")

from Load.load_text import load_text  # Directory loader for text files

files = load_text(directory)  # Loads data from directory


# 2. Split

print(f"\n\n\n2. Split:\n\n")

# 2.1 split generic files into smaller doc objects

from Setup.split_variables import chunk_size, chunk_overlap
from Split.split_files import split_files

textsplits = split_files(files, chunk_size, chunk_overlap)  # For Chroma as Pinecone splits in upsertion


# 3. Store

print("\n\n\n3. Store:\n\n")

# 3.1 create pinecone index

from Setup.pinecone_variables import env, index_name, pine_key
from Store.init_pinecone_index import create_index, initialize_pinecone

initialize_pinecone(pine_key, env)  # Initializes Pinecone

index = create_index(index_name, 1536, 'cosine')  # Creates index

# 3.2 upsert to pinecone index

from Setup.openai_variables import embed
from Store.upsert_to_pinecone import upsert_data_to_index

upsert_data_to_index(textsplits, embed, index)  # Upserts splits to Pinecone index

# 3.3 Create VectorDB's

from Store.create_pineconedb import create_pinecone_vectordb  # Pinecone DB

pinecone_vectordb = create_pinecone_vectordb(index_name, embed.embed_query)  # Initializes pinecone vectorstoredb

from Store.create_chromadb import create_chroma_vectordb  # Chroma DB

chroma_vectordb = create_chroma_vectordb(textsplits, embed)  # Initialize chroma and store docs


# 4. Retrieval

print("\n\n\n4. Retrieval:\n\n")

from Retrieval.pineconeretrieval import pineconeretrieval
from Retrieval.chromaretrieval import chromaretrieval

query = "H.E.R."  # Query to search for relevant docs

pineconeretriever = pineconeretrieval(pinecone_vectordb, query, "mmr")  # Initialize pinecone retriever with db and number of returned docs (k

chromaretriever = chromaretrieval(chroma_vectordb, query)  # Initialize chroma retriever with db and number of returned docs (k)


# # 5. Generate

# print("\n\n\n5. Generate:\n\n")

from Generate.retrievalqa_chain import create_retrievalqa_chain  # Retrieval QA Chain
from HelperFunctions.mainhelper import get_query
from Setup.openai_variables import llm


# 5.  Main Loop for Testing GQA Quality of LoadQA and RetrievalQA 

from HelperFunctions.generatehelper import configure_retrievalqa_chain

from collections import defaultdict

def main():
    ratings = {"Pinecone VectorDB": defaultdict(list),
               "Chroma VectorDB": defaultdict(list)}
    
    while True:
        query = get_query()
        if query is None:
            if any(ratings[db] for db in ratings):  # Check if there are any ratings
                # Flatten the ratings for easy computation
                flattened_ratings = {"{} - {}".format(db, method): ratings[db][method] for db in ratings for method in ratings[db]}
                highest_rated_method = max(flattened_ratings, key=lambda k: sum(flattened_ratings[k]) / len(flattened_ratings[k]))
                highest_rating = sum(flattened_ratings[highest_rated_method]) / len(flattened_ratings[highest_rated_method])
                
                print(f"\nThe highest rated method is '{highest_rated_method}' with an average rating of {highest_rating:.2f}.")
            else:
                print("\nNo ratings provided.")
            return
        
        else:
            while query:
                print("\nChoose a database to test:")
                print("1: Pinecone VectorDB + RetrievalQA Chain")
                print("2: Chroma VectorDB + LoadQA Chain")
                print("3: Go back to query selection")
                db_choice = input("\nEnter your Retrieval choice (1-3): ")
                if db_choice == "1":
                    retriever = pineconeretriever
                    chain_type, qa_chain = configure_retrievalqa_chain(create_retrievalqa_chain, llm, retriever)
                    db_type = "Pinecone VectorDB - " + chain_type
                elif db_choice == "2":
                    retriever = chromaretriever
                    chain_type, qa_chain = configure_retrievalqa_chain(create_retrievalqa_chain, llm, retriever)
                    db_type = "Chroma VectorDB - " + chain_type
                elif db_choice == "3":
                    break
                else:
                    print("Invalid choice. Please select between 1-3.")
                    continue
                if qa_chain:
                    answer = qa_chain.run(query)
                    print(f"\nChain Type: {db_type}")
                    print(f"Answer: {answer}")
                    while True:  # Loop to ensure valid input
                        try:
                            feedback = int(input("\nHow would you rate the quality of the answer (1-10)? "))
                            if 1 <= feedback <= 10:
                                ratings[db_type.split(" - ")[0]][chain_type].append(feedback)
                                print(f"Thank you for your feedback! You rated the answer a {feedback}.\n")
                                break  # Exit the loop once valid feedback is received
                            else:
                                print("Invalid rating. Please enter a number between 1 and 10.")
                        except ValueError:  # Handle non-integer inputs
                            print("Invalid input. Please enter a number between 1 and 10.")
                            
        
main()

index.delete(index_name)  # Delete index

chroma_vectordb.delete_collection()
chroma_vectordb.persist()