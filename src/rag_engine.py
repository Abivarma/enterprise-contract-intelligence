#!/usr/bin/env python3
"""
Retrieval-Augmented Generation (RAG) Engine for Enterprise Contract Intelligence.
Provides semantic search and context retrieval from contract documents.
"""

import os
import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class ContractChunk:
    """Represents a semantic chunk of a contract."""
    content: str
    chunk_id: int
    embedding: Optional[List[float]] = None
    metadata: Dict = None

    def to_dict(self):
        return asdict(self)


class VectorStore:
    """
    In-memory vector database for contract embeddings.
    
    For production, consider:
    - Pinecone: https://www.pinecone.io/
    - Weaviate: https://weaviate.io/
    - Milvus: https://milvus.io/
    - Chroma: https://www.trychroma.com/
    - FAISS: https://github.com/facebookresearch/faiss
    """

    def __init__(self, embedding_dim: int = 1536):
        """
        Initialize vector store.
        
        Args:
            embedding_dim: Dimension of embeddings (1536 for Ada-002)
        """
        self.embedding_dim = embedding_dim
        self.vectors: np.ndarray = None
        self.chunks: List[ContractChunk] = []
        self.chunk_count = 0

    def add_chunks(self, chunks: List[ContractChunk]) -> None:
        """
        Add multiple chunks with embeddings to store.
        
        Args:
            chunks: List of ContractChunk objects
        """
        if not chunks:
            logger.warning("No chunks to add to vector store")
            return

        embeddings = np.array([chunk.embedding for chunk in chunks])
        
        if self.vectors is None:
            self.vectors = embeddings
            self.chunks = chunks
        else:
            self.vectors = np.vstack([self.vectors, embeddings])
            self.chunks.extend(chunks)
        
        self.chunk_count = len(self.chunks)
        logger.info(f"Added {len(chunks)} chunks. Total chunks: {self.chunk_count}")

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[ContractChunk, float]]:
        """
        Semantic similarity search using cosine distance.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        if self.vectors is None or len(self.chunks) == 0:
            logger.warning("Vector store is empty")
            return []

        query_vec = np.array(query_embedding)
        
        # Normalize vectors for cosine similarity
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        vectors_norm = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-8)
        
        # Compute cosine similarities
        similarities = np.dot(vectors_norm, query_norm)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = [
            (self.chunks[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        logger.info(f"Retrieved {len(results)} similar chunks")
        return results

    def save(self, filepath: str) -> None:
        """
        Persist vector store to disk.
        
        Args:
            filepath: Path to save file
        """
        data = {
            'vectors': self.vectors.tolist() if self.vectors is not None else None,
            'chunks': [chunk.to_dict() for chunk in self.chunks],
            'embedding_dim': self.embedding_dim
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
        logger.info(f"Vector store saved to {filepath}")

    def load(self, filepath: str) -> None:
        """
        Load vector store from disk.
        
        Args:
            filepath: Path to load file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.embedding_dim = data['embedding_dim']
        if data['vectors']:
            self.vectors = np.array(data['vectors'])
        
        self.chunks = [
            ContractChunk(
                content=chunk['content'],
                chunk_id=chunk['chunk_id'],
                embedding=chunk.get('embedding'),
                metadata=chunk.get('metadata', {})
            )
            for chunk in data['chunks']
        ]
        self.chunk_count = len(self.chunks)
        logger.info(f"Vector store loaded with {self.chunk_count} chunks")


class RAGEngine:
    """
    Retrieval-Augmented Generation Engine for contract analysis.
    
    Components:
    1. Document Chunking: Splits contracts into semantic chunks
    2. Embedding Generation: Creates vector representations
    3. Vector Store: Stores and searches embeddings
    4. Retrieval: Fetches relevant context for queries
    5. Generation: Uses retrieved context for LLM responses
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize RAG Engine.
        
        Args:
            chunk_size: Size of text chunks in tokens
            chunk_overlap: Overlap between consecutive chunks
        """
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=self.api_key)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = VectorStore()
        logger.info("RAG Engine initialized")

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Strategy: Split by sentences/paragraphs, then enforce max size.
        
        Args:
            text: Input text
            
        Returns:
            List of text chunks
        """
        # Split by double newlines first (paragraphs)
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) <= self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        logger.info(f"Text chunked into {len(chunks)} chunks")
        return chunks

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate OpenAI embeddings for texts.
        
        Uses Ada-002 model for best performance/cost ratio.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i, text in enumerate(texts):
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=text
                )
                embedding = response.data[0].embedding
                embeddings.append(embedding)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Generated embeddings for {i + 1}/{len(texts)} texts")
                    
            except Exception as e:
                logger.error(f"Error generating embedding for text {i}: {str(e)}")
                # Use zero vector as fallback
                embeddings.append([0.0] * 1536)
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings

    def ingest_contract(self, contract_text: str) -> None:
        """
        Ingest contract into RAG system.
        
        Process:
        1. Chunk the contract
        2. Generate embeddings
        3. Store in vector database
        
        Args:
            contract_text: Full contract text
        """
        logger.info("Starting contract ingestion...")
        
        # Step 1: Chunk
        chunks = self.chunk_text(contract_text)
        logger.info(f"Created {len(chunks)} chunks from contract")
        
        # Step 2: Generate embeddings
        embeddings = self.generate_embeddings(chunks)
        
        # Step 3: Create ContractChunk objects
        contract_chunks = [
            ContractChunk(
                content=chunk,
                chunk_id=i,
                embedding=embedding,
                metadata={'source': 'contract', 'chunk_index': i}
            )
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        
        # Step 4: Store in vector store
        self.vector_store.add_chunks(contract_chunks)
        logger.info("Contract ingestion completed")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve relevant contract sections for query.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of relevant chunks with metadata
        """
        logger.info(f"Retrieving relevant chunks for query: {query[:50]}...")
        
        # Generate query embedding
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        )
        query_embedding = response.data[0].embedding
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k)
        
        # Format results
        formatted_results = [
            {
                'content': chunk.content,
                'chunk_id': chunk.chunk_id,
                'similarity_score': float(score),
                'metadata': chunk.metadata
            }
            for chunk, score in results
        ]
        
        logger.info(f"Retrieved {len(formatted_results)} relevant chunks")
        return formatted_results

    def augmented_query(self, query: str, system_prompt: str, top_k: int = 5) -> str:
        """
        Generate LLM response using retrieved context (RAG).
        
        This is the "Augmented Generation" part of RAG.
        
        Args:
            query: User query
            system_prompt: System prompt for LLM
            top_k: Number of context chunks to retrieve
            
        Returns:
            LLM response with context
        """
        logger.info("Starting augmented generation...")
        
        # Step 1: Retrieve relevant context
        retrieved = self.retrieve(query, top_k)
        context = "\n\n".join([r['content'] for r in retrieved])
        
        # Step 2: Create augmented prompt
        augmented_prompt = f"""
Context from contract:
{context}

Question: {query}

Provide a precise answer based on the above contract context.
"""
        
        # Step 3: Call LLM with context
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": augmented_prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            
            result = response.choices[0].message.content
            logger.info("Augmented generation completed")
            return result
            
        except Exception as e:
            logger.error(f"Error in augmented generation: {str(e)}")
            raise

    def get_vector_store_info(self) -> Dict:
        """
        Get information about current vector store.
        
        Returns:
            Dictionary with vector store statistics
        """
        return {
            'total_chunks': self.vector_store.chunk_count,
            'embedding_dimension': self.vector_store.embedding_dim,
            'vectors_stored': self.vector_store.vectors is not None
        }
