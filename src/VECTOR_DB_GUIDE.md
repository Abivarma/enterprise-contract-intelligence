# Vector Database & RAG Implementation Guide

## Overview

This guide explains the Retrieval-Augmented Generation (RAG) system and Vector Database implementation in the Enterprise Contract Intelligence Platform.

## Table of Contents
1. [RAG Architecture](#rag-architecture)
2. [Vector Database Implementation](#vector-database-implementation)
3. [Usage Examples](#usage-examples)
4. [Production Deployment](#production-deployment)
5. [Performance Tuning](#performance-tuning)

---

## RAG Architecture

### What is RAG?

Retrieval-Augmented Generation (RAG) combines:
- **Retrieval**: Fetching relevant documents/chunks from a knowledge base
- **Augmentation**: Adding retrieved context to the LLM prompt
- **Generation**: Using LLM to generate responses based on context

### Why RAG for Contracts?

1. **Accuracy**: Grounds LLM responses in actual contract text
2. **Reduce Hallucinations**: Prevents AI from making up information
3. **Legal Compliance**: Ensures responses reference specific clauses
4. **Audit Trail**: Can cite which section was used for each decision

### RAG Pipeline

```
Contract Input
    ↓
Document Chunking (500-token chunks)
    ↓
Embedding Generation (OpenAI Ada-002)
    ↓
Vector Store (In-memory or external DB)
    ↓
User Query
    ↓
Generate Query Embedding
    ↓
Semantic Search (Cosine Similarity)
    ↓
Retrieve Top-K Relevant Chunks
    ↓
Augment LLM Prompt with Context
    ↓
Generate Response
    ↓
Cited Answer
```

---

## Vector Database Implementation

### Current Implementation: In-Memory VectorStore

The `VectorStore` class in `rag_engine.py` provides:

#### Features
- **In-memory storage** using NumPy arrays
- **Cosine similarity search** for semantic retrieval
- **JSON persistence** (save/load to disk)
- **Fast retrieval** (milliseconds for 1000+ chunks)

#### Code Example

```python
from src.rag_engine import VectorStore, ContractChunk

# Initialize
vector_store = VectorStore(embedding_dim=1536)

# Add chunks with embeddings
chunks = [
    ContractChunk(
        content="Confidentiality clause: ...",
        chunk_id=0,
        embedding=[0.1, 0.2, ..., 0.9],  # 1536-dim vector
        metadata={'section': 'Confidentiality', 'page': 1}
    ),
    # ... more chunks
]
vector_store.add_chunks(chunks)

# Search
query_embedding = [0.1, 0.15, ..., 0.85]  # Query vector
results = vector_store.search(query_embedding, top_k=5)

for chunk, score in results:
    print(f"Score: {score:.3f}")
    print(f"Content: {chunk.content[:100]}...")

# Persist
vector_store.save('vector_store.json')

# Load
vector_store.load('vector_store.json')
```

### Production-Grade Vector Databases

For production deployments, consider:

#### 1. **Pinecone**
```
Pros:
- Fully managed, serverless
- Auto-scaling
- 99.95% uptime SLA
- 1M+ documents support

Cons:
- Cloud-only (no on-premise)
- Pay-per-query pricing

Integration:
pip install pinecone-client
```

#### 2. **Weaviate**
```
Pros:
- Open-source or cloud
- On-premise capable
- Rich query language
- Hybrid search (vector + keyword)

Cons:
- Requires infrastructure management
- Steeper learning curve

Integration:
pip install weaviate-client
```

#### 3. **Milvus**
```
Pros:
- Open-source
- Extremely fast (GPU support)
- 100M+ vectors support
- Self-hosted

Cons:
- Requires Kubernetes/Docker
- More operational overhead

Integration:
pip install pymilvus
```

#### 4. **FAISS (Facebook AI Similarity Search)**
```
Pros:
- Lightweight, no dependencies
- GPU-accelerated
- Excellent for local development
- Billions of vectors support

Cons:
- Basic query interface
- Limited filtering capabilities

Integration:
pip install faiss-cpu  # or faiss-gpu
```

#### 5. **Chroma**
```
Pros:
- Simplest to get started
- Persistent client
- Built for LLM applications
- Lightweight

Cons:
- Newer (less battle-tested)
- Limited scalability

Integration:
pip install chromadb
```

---

## Usage Examples

### Basic RAG Pipeline

```python
from src.rag_engine import RAGEngine
import os

# Initialize RAG Engine
rag = RAGEngine(chunk_size=500, chunk_overlap=50)

# Read contract
with open('contracts/sample_contract.txt', 'r') as f:
    contract_text = f.read()

# Ingest contract
rag.ingest_contract(contract_text)

# Retrieve relevant chunks
query = "What are the payment terms?"
results = rag.retrieve(query, top_k=5)

for result in results:
    print(f"Similarity: {result['similarity_score']:.3f}")
    print(f"Content: {result['content'][:200]}...\n")
```

### Augmented Generation

```python
# Generate response using RAG context
system_prompt = "You are a contract analysis expert. Provide precise answers based on the contract context."

response = rag.augmented_query(
    query="What are the termination conditions?",
    system_prompt=system_prompt,
    top_k=5
)

print(response)
# Output will cite the specific contract sections
```

### Vector Store Statistics

```python
info = rag.get_vector_store_info()
print(f"Total chunks: {info['total_chunks']}")
print(f"Embedding dimension: {info['embedding_dimension']}")
print(f"Vectors stored: {info['vectors_stored']}")
```

---

## Production Deployment

### Scaling Considerations

| Aspect | In-Memory | Pinecone | Weaviate | Milvus |
|--------|-----------|----------|----------|--------|
| **Max Documents** | 100K | 1M+ | 1M+ | 100M+ |
| **Query Latency** | <10ms | 100-500ms | 50-200ms | <50ms |
| **Setup Time** | <1min | 5min | 1hour | 2hours |
| **Monthly Cost** | $0 | $100-1000 | $0 | $0 |
| **Uptime SLA** | N/A | 99.95% | 99.9% | 99.95% |

### Migration from In-Memory to Pinecone

```python
from pinecone import Pinecone

# Initialize Pinecone
pc = Pinecone(api_key="your-api-key")
index = pc.Index("contracts")

# Export from in-memory
vectors_to_upsert = [
    (str(chunk.chunk_id), chunk.embedding, chunk.metadata)
    for chunk in rag.vector_store.chunks
]

# Import to Pinecone
index.upsert(vectors=vectors_to_upsert)
```

---

## Performance Tuning

### Chunk Size Optimization

```
Small chunks (100-300 tokens):
- ✓ More precise retrieval
- ✓ Lower latency
- ✗ More vectors to store
- ✗ May lose context

Medium chunks (300-800 tokens):
- ✓ Good balance
- ✓ Optimal for contracts
- ✓ Industry standard

Large chunks (800+ tokens):
- ✓ Better context preservation
- ✗ Less precise retrieval
- ✗ Higher latency
```

### Embedding Model Selection

```
OpenAI Ada-002 (Recommended):
- 1536 dimensions
- $0.02 per 1M tokens
- 95%+ accuracy
- 130K token context

OpenAI 3-Small:
- 1536 dimensions
- $0.02 per 1M tokens
- Similar performance to Ada

Mistral Embed:
- 1024 dimensions
- $0.10 per 1M tokens
- Good for legal domain
```

### Search Parameters

```python
# top_k: Number of results
# Recommendations:
top_k = 3   # Quick analysis, specific queries
top_k = 5   # Balanced (default)
top_k = 10  # Comprehensive review, complex queries

# Similarity threshold
min_score = 0.7  # Only return highly relevant chunks
```

---

## Cost Analysis

### Monthly Cost Breakdown (10,000 contracts)

| Component | Cost |
|-----------|------|
| **Embeddings** (Ada-002) | $20 |
| **LLM Queries** (GPT-3.5) | $50 |
| **Vector DB** (In-memory) | $0 |
| **Storage** (GitHub) | $0 |
| **Total** | **$70** |

*vs. Manual Review: $24,000-50,000/month*

---

## Troubleshooting

### Poor Retrieval Results

1. **Check chunk size**: Too large/small?
2. **Verify embeddings**: Are they being generated correctly?
3. **Inspect similarity scores**: All <0.5 means irrelevant chunks
4. **Increase top_k**: Retrieve more candidates

### Slow Queries

1. **Reduce chunk count**: Delete unnecessary vectors
2. **Use smaller top_k**: Balance speed vs. comprehensiveness
3. **Switch to production DB**: In-memory slows down with >100K chunks

### Memory Issues

1. **Migrate to external vector DB**: Pinecone, Weaviate, etc.
2. **Use vector compression**: Reduce embedding dimensions
3. **Implement chunking**: Process contracts in batches

---

## References

- [RAG Research Paper](https://arxiv.org/abs/2005.11401)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Weaviate Docs](https://weaviate.io/developers/weaviate)
- [LangChain RAG](https://python.langchain.com/docs/use_cases/question_answering)
