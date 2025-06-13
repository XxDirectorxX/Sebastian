
# core/memory/vector_store.py

import os
import logging
from chromadb import Client
from chromadb.config import Settings

logger = logging.getLogger("Sebastian.VectorStore")

client = None

def init_vector_store(persist_directory: str = "core/memory/vector_data"):
    global client
    try:
        client = Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory
        ))
        logger.info(f"[VectorStore] Initialized at: {persist_directory}")
    except Exception as e:
        logger.exception("[VectorStore] Failed to initialize Chroma vector store")

def get_collection(name: str = "sebastian_memory"):
    if not client:
        raise RuntimeError("[VectorStore] Client not initialized. Call init_vector_store() first.")
    return client.get_or_create_collection(name=name)
