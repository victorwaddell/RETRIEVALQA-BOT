# Splitfiles: Split files into text splits

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_docs(documents, chunk_size, chunk_overlap):  # Splits documents into text splits
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    splits = []
    for document in documents:
        source = document.metadata.get('source', "DEFAULT_SOURCE")
        if source == "DEFAULT_SOURCE":
            print(f"Document {document} did not have a source. Setting default source.")
        source = os.path.splitext(os.path.basename(document.metadata['source']))[0]
        splits_for_document = text_splitter.split_documents([document])
        splits.extend([{  # Adds split + metadata to splits list
            'id': document.metadata['source'] + f'-{i}',
            'text': split.page_content,
            'source': source,
            'chunk': i
        } for i, split in enumerate(splits_for_document)])
    seen = set()  # Remove redundant splits based on text content
    new_splits = []  
    for split in splits:  
        text = split['text']  
        if text not in seen:  # If text is not in seen, add to new splits
            new_splits.append(split)
            seen.add(text)
    print(f"Split {len(splits)} documents into {len(new_splits)} text splits!\n")
    return new_splits