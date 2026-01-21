#!/usr/bin/env python3
"""
Enterprise Contract Intelligence Platform
Main module for contract analysis using OpenAI API and RAG patterns.
"""

import os
import json
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI
from loguru import logger

# Load environment variables
load_dotenv()

# Configure logging
logger.add(
    "logs/contract_analysis.log",
    rotation="500 MB",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
)

class ContractAnalyzer:
    """Contract analysis engine using OpenAI API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the contract analyzer."""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError('OPENAI_API_KEY environment variable not set')

        # Remove invalid OPENAI_ORG_ID if it exists (common mistake in .env files)
        org_id = os.getenv('OPENAI_ORG_ID')
        if org_id and org_id.lower() in ['personal', 'your_org_id_here', 'none']:
            logger.warning(f'Ignoring invalid OPENAI_ORG_ID: {org_id}')
            if 'OPENAI_ORG_ID' in os.environ:
                del os.environ['OPENAI_ORG_ID']

        self.client = OpenAI(api_key=self.api_key)
        self.model = os.getenv('MODEL_NAME', 'gpt-3.5-turbo')
        logger.info(f'ContractAnalyzer initialized with model: {self.model}')

    def analyze_contract(self, contract_text: str) -> dict:
        """Analyze contract and extract key information."""
        logger.info('Starting contract analysis')
        
        system_prompt = """
You are an expert contract analyst with 20+ years of experience in legal document analysis.
Analyze the provided contract and extract key information.
Provide structured analysis including:
1. Contract type and parties
2. Key dates and durations
3. Financial terms
4. Risk factors
5. Compliance requirements
6. Termination clauses
7. Liability and indemnification
8. Overall risk assessment (Low/Medium/High)
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this contract:\n\n{contract_text}"}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            analysis_text = response.choices[0].message.content
            logger.info('Contract analysis completed successfully')
            
            return {
                "status": "success",
                "analysis": analysis_text,
                "model": self.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            error_msg = str(e)
            logger.error(f'Error during contract analysis: {error_msg}')

            # Add helpful context for common errors
            if "403" in error_msg or "PermissionDenied" in error_msg:
                error_msg += " - Check your API key validity and billing status at https://platform.openai.com/account/billing"
            elif "401" in error_msg:
                error_msg += " - Invalid API key. Please check OPENAI_API_KEY in .env"

            return {
                "status": "error",
                "error": error_msg
            }

    def assess_risk(self, contract_text: str) -> dict:
        """Assess risk level of contract."""
        logger.info('Starting risk assessment')
        
        system_prompt = """
You are a risk assessment expert for legal contracts.
Evaluate the contract for potential risks and provide a risk score (1-10).
Identify critical risk areas that require legal review.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Assess risks in this contract:\n\n{contract_text}"}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            assessment = response.choices[0].message.content
            logger.info('Risk assessment completed')
            
            return {
                "status": "success",
                "assessment": assessment
            }
        except Exception as e:
            error_msg = str(e)
            logger.error(f'Error during risk assessment: {error_msg}')

            # Add helpful context for common errors
            if "403" in error_msg or "PermissionDenied" in error_msg:
                error_msg += " - Check your API key validity and billing status at https://platform.openai.com/account/billing"
            elif "401" in error_msg:
                error_msg += " - Invalid API key. Please check OPENAI_API_KEY in .env"

            return {"status": "error", "error": error_msg}


def main():
    """Main execution function."""
    logger.info('Starting Enterprise Contract Intelligence Platform')
    
    # Example usage - can be modified for file processing
    sample_contract = """Sample contract text for analysis..."""
    
    try:
        analyzer = ContractAnalyzer()
        results = analyzer.analyze_contract(sample_contract)
        logger.info(f'Analysis results: {results}')
        print(json.dumps(results, indent=2))
    except Exception as e:
        logger.error(f'Fatal error: {str(e)}')
        raise


if __name__ == "__main__":
    main()
