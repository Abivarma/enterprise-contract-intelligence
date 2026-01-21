#!/usr/bin/env python3
"""
RAG Demo Execution Script
Runs RAG pipeline on sample contracts and saves results with visualizations.
Designed to run locally or in GitHub Actions CI/CD.
"""

import os
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

try:
    from src.rag_engine import RAGEngine
except ImportError:
    print("Error: Could not import RAGEngine. Make sure src/rag_engine.py exists.")
    sys.exit(1)

# Configure paths
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "rag_results"
VIZUALIZATION_DIR = BASE_DIR / "rag_visualizations"
CONTRACTS_DIR = BASE_DIR / "sample_contracts"

# Create directories
RESULTS_DIR.mkdir(exist_ok=True)
VIZUALIZATION_DIR.mkdir(exist_ok=True)
CONTRACTS_DIR.mkdir(exist_ok=True)


def create_sample_contracts():
    """
    Create sample contracts for demonstration.
    """
    print("\n[*] Creating sample contracts...")
    
    sample_contract_1 = """CONFIDENTIAL AGREEMENT

This Confidentiality Agreement ("Agreement") is entered into as of January 1, 2024,
between ABC Corporation ("Company") and XYZ Services Inc. ("Recipient").

1. CONFIDENTIAL INFORMATION

The Company agrees to disclose certain confidential information to the Recipient,
including but not limited to:
- Technical specifications and source code
- Business plans and financial projections
- Customer lists and pricing information
- Proprietary algorithms and methodologies

2. OBLIGATIONS OF RECIPIENT

The Recipient agrees to:
- Keep all confidential information strictly confidential
- Use the information only for the purposes outlined in this agreement
- Protect the information with reasonable security measures
- Not disclose the information to third parties without written consent

3. TERM AND TERMINATION

This Agreement shall remain in effect for a period of 3 years from the date of disclosure.
Either party may terminate this agreement with 30 days written notice.

4. GOVERNING LAW

This Agreement shall be governed by the laws of the State of California.

5. PAYMENT TERMS

The Company shall pay the Recipient a service fee of $50,000 per month.
Payment is due within 15 days of invoice receipt.
Late payments will incur a 2% monthly penalty.
"""
    
    sample_contract_2 = """SERVICE AGREEMENT

This Service Agreement is entered into on February 15, 2024.

1. SERVICES PROVIDED

ServiceProvider agrees to provide the following services:
- 24/7 technical support
- Monthly maintenance and updates
- Performance monitoring and reporting
- Annual security audits

2. PRICING AND PAYMENT

Annual Service Fee: $120,000
- First payment due upon execution: $30,000
- Quarterly payments: $22,500
- Payment method: Bank transfer or credit card
- Discounts available for 2+ year contracts

3. LIABILITY LIMITATIONS

ServiceProvider's total liability shall not exceed the fees paid in the preceding 12 months.
ServiceProvider is not liable for:
- Indirect or consequential damages
- Loss of revenue or business opportunities
- Data loss not caused by ServiceProvider negligence

4. TERMINATION

This agreement can be terminated:
- For convenience with 60 days written notice
- For cause immediately if either party breaches material terms
- Automatically after 3 years unless renewed

5. INTELLECTUAL PROPERTY

All work product developed by ServiceProvider remains ServiceProvider's property.
Client receives a non-exclusive license to use the work product.
"""
    
    contract_path_1 = CONTRACTS_DIR / "sample_contract_1.txt"
    contract_path_2 = CONTRACTS_DIR / "sample_contract_2.txt"
    
    contract_path_1.write_text(sample_contract_1)
    contract_path_2.write_text(sample_contract_2)
    
    print(f"✓ Created {contract_path_1}")
    print(f"✓ Created {contract_path_2}")
    
    return [contract_path_1, contract_path_2]


def run_rag_pipeline(contract_paths: List[Path]) -> Dict:
    """
    Run RAG pipeline on sample contracts.
    """
    print("\n[*] Initializing RAG Engine...")
    
    try:
        rag = RAGEngine(chunk_size=300, chunk_overlap=50)
    except ValueError as e:
        print(f"✗ Error initializing RAG Engine: {e}")
        print("  Make sure OPENAI_API_KEY is set in environment variables.")
        return {"status": "error", "message": str(e)}
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "contracts_analyzed": [],
        "queries_executed": [],
        "statistics": {}
    }
    
    # Ingest contracts
    for contract_path in contract_paths:
        print(f"\n[*] Processing: {contract_path.name}")
        
        contract_text = contract_path.read_text()
        
        try:
            rag.ingest_contract(contract_text)
            
            # Get vector store info
            info = rag.get_vector_store_info()
            
            contract_result = {
                "filename": contract_path.name,
                "size_bytes": len(contract_text),
                "chunks_created": info['total_chunks'],
                "embedding_dimension": info['embedding_dimension']
            }
            results["contracts_analyzed"].append(contract_result)
            
            print(f"✓ Chunks created: {info['total_chunks']}")
            print(f"✓ Embedding dimension: {info['embedding_dimension']}")
            
        except Exception as e:
            print(f"✗ Error processing contract: {e}")
            continue
    
    # Execute queries
    queries = [
        "What are the payment terms?",
        "What is the termination policy?",
        "What confidentiality obligations exist?",
        "What is the liability limitation?",
        "What services are provided?"
    ]
    
    print("\n[*] Executing RAG queries...")
    
    for i, query in enumerate(queries, 1):
        print(f"\n  Query {i}/5: {query}")
        
        try:
            # Retrieve relevant chunks
            retrieved = rag.retrieve(query, top_k=3)
            
            # Generate augmented response
            system_prompt = "You are a contract analysis expert. Answer based on the provided contract context."
            response = rag.augmented_query(query, system_prompt, top_k=3)
            
            query_result = {
                "query": query,
                "retrieved_chunks": len(retrieved),
                "top_chunk_score": retrieved[0]['similarity_score'] if retrieved else 0,
                "response_preview": response[:200] + "..." if len(response) > 200 else response
            }
            results["queries_executed"].append(query_result)
            
            print(f"  ✓ Retrieved {len(retrieved)} relevant chunks")
            print(f"  ✓ Top relevance score: {query_result['top_chunk_score']:.3f}")
            
        except Exception as e:
            print(f"  ✗ Error executing query: {e}")
            continue
    
    results["statistics"] = {
        "total_contracts_processed": len(results["contracts_analyzed"]),
        "total_queries_executed": len(results["queries_executed"]),
        "average_relevance_score": round(
            sum(q.get('top_chunk_score', 0) for q in results["queries_executed"]) / 
            max(len(results["queries_executed"]), 1), 3
        ),
        "status": "success"
    }
    
    return results


def save_results(results: Dict) -> Path:
    """
    Save RAG results to JSON file.
    """
    print("\n[*] Saving results...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"rag_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {results_file}")
    return results_file


def create_visualization(results: Dict) -> Path:
    """
    Create a visual markdown report of RAG results.
    """
    print("\n[*] Creating visualization report...")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    md_content = f"""# RAG Execution Results

**Generated:** {timestamp}

---

## 📊 Summary Statistics

| Metric | Value |
|--------|-------|
| Contracts Processed | {results['statistics']['total_contracts_processed']} |
| Queries Executed | {results['statistics']['total_queries_executed']} |
| Avg Relevance Score | {results['statistics']['average_relevance_score']} |
| Status | {results['statistics']['status']} |

---

## 📁 Contracts Analyzed

"""
    
    for contract in results["contracts_analyzed"]:
        md_content += f"""\n### {contract['filename']}
- **Size:** {contract['size_bytes']:,} bytes
- **Chunks Created:** {contract['chunks_created']}
- **Embedding Dimension:** {contract['embedding_dimension']}
"""
    
    md_content += """\n---

## 🔍 Query Results

"""
    
    for i, query_result in enumerate(results["queries_executed"], 1):
        md_content += f"""\n### Query {i}: {query_result['query']}

- **Retrieved Chunks:** {query_result['retrieved_chunks']}
- **Top Chunk Relevance Score:** {query_result['top_chunk_score']:.3f}
- **Response Preview:**
  > {query_result['response_preview']}

"""
    
    md_content += f"""\n---

## 📈 Performance Metrics

```
Average Relevance Score: {results['statistics']['average_relevance_score']}
Total Embeddings Generated: {sum(c['chunks_created'] for c in results['contracts_analyzed'])}
Queries Processed: {results['statistics']['total_queries_executed']}
Execution Timestamp: {timestamp}
```

---

**Last Updated:** {timestamp}
"""
    
    viz_file = VISUALIZATION_DIR / f"rag_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(viz_file, 'w') as f:
        f.write(md_content)
    
    print(f"✓ Visualization saved to: {viz_file}")
    return viz_file


def main():
    """
    Main execution function.
    """
    print("="*60)
    print("RAG EXECUTION DEMO")
    print("Enterprise Contract Intelligence Platform")
    print("="*60)
    
    try:
        # Step 1: Create sample contracts
        contract_paths = create_sample_contracts()
        
        # Step 2: Run RAG pipeline
        results = run_rag_pipeline(contract_paths)
        
        if results.get("status") == "error":
            print(f"\n✗ Pipeline failed: {results.get('message')}")
            return 1
        
        # Step 3: Save results
        results_file = save_results(results)
        
        # Step 4: Create visualization
        viz_file = create_visualization(results)
        
        # Step 5: Print summary
        print("\n" + "="*60)
        print("EXECUTION SUMMARY")
        print("="*60)
        print(f"✓ Contracts processed: {results['statistics']['total_contracts_processed']}")
        print(f"✓ Queries executed: {results['statistics']['total_queries_executed']}")
        print(f"✓ Avg relevance score: {results['statistics']['average_relevance_score']}")
        print(f"✓ Results file: {results_file}")
        print(f"✓ Visualization: {viz_file}")
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
