import time
import openai
from tqdm.auto import tqdm

def create_embeddings(texts, embed_model):  # Creates embeddings for texts
    try:  # Tries to create embeddings
        return openai.Embedding.create(input = texts, engine = embed_model)
    except Exception as e:  # Prints error if embedding fails
        print(f"Rate limit or other exception hit during embedding: {e}")
        time.sleep(5)
        return create_embeddings(texts, embed_model)  # Retries embedding

def upsert_data_to_index(splits, batch_size, embed_model, index):  # Upserts data to Pinecone index
    for i in tqdm(range(0, len(splits), batch_size)):  # Iterates through splits
        i_end = min(len(splits), i + batch_size)  # Gets end of batch
        meta_batch = splits[i:i_end]
        ids_batch = [x['id'] for x in meta_batch]  # Gets ids, texts, and embeddings for batch
        texts = [x['text'] for x in meta_batch]
        res = create_embeddings(texts, embed_model)
        embeds = [record['embedding'] for record in res['data']]
        to_upsert = list(zip(ids_batch, embeds, meta_batch))  # Zips ids, embeddings, and metadata
        try:  # Tries to upsert data
            index.upsert(vectors = to_upsert)
        except Exception as e:  # Prints error if upsert fails
            print(f"Error during Pinecone upsert at batch starting at index {i}: {e}\nProblematic batch: {meta_batch}")
