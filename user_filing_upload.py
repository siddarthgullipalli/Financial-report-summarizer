"""
User SEC Filing Upload and Comparative Analysis Module

This module extends the FinBERTFinancialRAG system to allow users to:
1. Upload their own SEC filings (PDF, TXT, or extracted text)
2. Process and index them in ChromaDB alongside existing filings
3. Ask comparative questions across their filing and other companies
"""

import os
import re
import uuid
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import PyPDF2
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb


class UserFilingManager:
    """
    Manages user-uploaded SEC filings for comparative analysis
    """

    def __init__(self, rag_instance):
        """
        Initialize the manager with an existing RAG instance

        Args:
            rag_instance: Instance of FinBERTFinancialRAG class
        """
        self.rag = rag_instance
        self.embedder = rag_instance.embedder
        self.collection = rag_instance.collection
        self.user_uploads_dir = Path("user_uploads")
        self.user_uploads_dir.mkdir(exist_ok=True)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text content
        """
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

        return text

    def extract_text_from_txt(self, txt_path: str) -> str:
        """
        Extract text from a TXT file

        Args:
            txt_path: Path to the text file

        Returns:
            File content
        """
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            raise Exception(f"Error reading text file: {str(e)}")

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """
        Split text into overlapping chunks

        Args:
            text: Input text to chunk
            chunk_size: Target size of each chunk in characters
            overlap: Number of overlapping characters between chunks

        Returns:
            List of text chunks
        """
        # Split by paragraphs first
        paragraphs = text.split('\n\n')

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If paragraph is very long, split it
            if len(para) > chunk_size:
                words = para.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk) + len(word) + 1 <= chunk_size:
                        temp_chunk += word + " "
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = word + " "
                if temp_chunk:
                    chunks.append(temp_chunk.strip())
            else:
                # Try to add paragraph to current chunk
                if len(current_chunk) + len(para) + 2 <= chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        # Apply overlap
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i > 0 and overlap > 0:
                # Add overlap from previous chunk
                prev_chunk = chunks[i-1]
                overlap_text = prev_chunk[-overlap:] if len(prev_chunk) > overlap else prev_chunk
                overlapped_chunks.append(overlap_text + " " + chunk)
            else:
                overlapped_chunks.append(chunk)

        # Filter out very short chunks
        return [chunk for chunk in overlapped_chunks if len(chunk) > 50]

    def upload_user_filing(
        self,
        file_path: str,
        company_name: str,
        filing_type: str = "10-K",
        fiscal_year: Optional[str] = None,
        section_name: str = "Full Filing",
        cik: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Upload and process a user's SEC filing

        Args:
            file_path: Path to the filing (PDF or TXT)
            company_name: Name of the company (user's company)
            filing_type: Type of filing (10-K, 10-Q, 8-K, etc.)
            fiscal_year: Fiscal year of the filing
            section_name: Section name (e.g., "MD&A", "Business Description")
            cik: Central Index Key (optional)

        Returns:
            Dictionary with upload statistics
        """
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Extract text based on file type
        file_ext = Path(file_path).suffix.lower()
        print(f"📄 Processing {file_ext} file: {file_path}")

        if file_ext == '.pdf':
            text = self.extract_text_from_pdf(file_path)
        elif file_ext in ['.txt', '.text']:
            text = self.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}. Use PDF or TXT files.")

        if not text.strip():
            raise ValueError("Extracted text is empty. Please check the file.")

        print(f"✅ Extracted {len(text)} characters")

        # Chunk the text
        chunks = self.chunk_text(text)
        print(f"✂️  Created {len(chunks)} chunks")

        # Generate embeddings
        print("🧠 Generating embeddings...")
        embeddings = self.embedder.encode(
            chunks,
            device='cuda' if self.rag.device == 'cuda' else 'cpu',
            show_progress_bar=True,
            batch_size=8
        )

        # Prepare metadata
        fiscal_year = fiscal_year or datetime.now().year
        cik = cik or f"USER_{uuid.uuid4().hex[:8].upper()}"

        metadatas = []
        ids = []

        for i in range(len(chunks)):
            metadata = {
                'company': company_name,
                'filing_type': filing_type,
                'filing_date': str(fiscal_year),
                'section': section_name,
                'cik': cik,
                'year': str(fiscal_year),
                'index': str(i),
                'source': 'user_upload',
                'upload_timestamp': datetime.now().isoformat()
            }
            metadatas.append(metadata)
            ids.append(f"user_{cik}_{fiscal_year}_{section_name}_{i}")

        # Add to ChromaDB
        print("💾 Adding to vector database...")
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )

        print(f"✅ Successfully uploaded {company_name} filing!")

        return {
            'company_name': company_name,
            'filing_type': filing_type,
            'fiscal_year': fiscal_year,
            'cik': cik,
            'chunks_created': len(chunks),
            'total_characters': len(text),
            'section': section_name
        }

    def get_user_uploaded_companies(self) -> List[Dict[str, str]]:
        """
        Get list of all user-uploaded companies

        Returns:
            List of dictionaries with company information
        """
        # Get all documents from collection
        all_docs = self.collection.get()

        # Filter user uploads and extract unique companies
        user_companies = {}

        if all_docs and all_docs['metadatas']:
            for metadata in all_docs['metadatas']:
                if metadata.get('source') == 'user_upload':
                    cik = metadata.get('cik')
                    if cik and cik not in user_companies:
                        user_companies[cik] = {
                            'company_name': metadata.get('company'),
                            'cik': cik,
                            'filing_type': metadata.get('filing_type'),
                            'fiscal_year': metadata.get('year'),
                            'upload_timestamp': metadata.get('upload_timestamp')
                        }

        return list(user_companies.values())

    def delete_user_filing(self, cik: str) -> bool:
        """
        Delete a user-uploaded filing from the database

        Args:
            cik: CIK of the company to delete

        Returns:
            True if successful
        """
        # Get all documents
        all_docs = self.collection.get()

        # Find IDs to delete
        ids_to_delete = []

        if all_docs and all_docs['metadatas']:
            for i, metadata in enumerate(all_docs['metadatas']):
                if (metadata.get('source') == 'user_upload' and
                    metadata.get('cik') == cik):
                    ids_to_delete.append(all_docs['ids'][i])

        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            print(f"🗑️  Deleted {len(ids_to_delete)} chunks for CIK: {cik}")
            return True
        else:
            print(f"⚠️  No user uploads found for CIK: {cik}")
            return False


class ComparativeAnalyzer:
    """
    Performs comparative analysis between user's filing and other companies
    """

    def __init__(self, rag_instance):
        """
        Initialize with RAG instance

        Args:
            rag_instance: Instance of FinBERTFinancialRAG
        """
        self.rag = rag_instance
        self.collection = rag_instance.collection

    def ask_comparative_question(
        self,
        question: str,
        user_company: str,
        comparison_companies: Optional[List[str]] = None,
        top_k: int = 10,
        use_hybrid: bool = True
    ) -> Dict[str, any]:
        """
        Ask a comparative question about user's filing vs other companies

        Args:
            question: The comparative question
            user_company: Name of user's uploaded company
            comparison_companies: List of companies to compare with (None for all)
            top_k: Number of chunks to retrieve per company
            use_hybrid: Whether to use hybrid search

        Returns:
            Dictionary with answer and supporting evidence
        """
        # Retrieve context for user's company
        print(f"🔍 Retrieving context for {user_company}...")
        user_context = self._get_company_context(
            question, user_company, top_k, use_hybrid
        )

        # Retrieve context for comparison companies
        comparison_contexts = {}

        if comparison_companies:
            for company in comparison_companies:
                print(f"🔍 Retrieving context for {company}...")
                comparison_contexts[company] = self._get_company_context(
                    question, company, top_k, use_hybrid
                )
        else:
            # Get context from all companies (limited)
            print("🔍 Retrieving context from all companies...")
            all_context = self._get_general_context(question, top_k * 2, use_hybrid)

            # Organize by company
            for chunk, metadata in zip(all_context['chunks'], all_context['metadatas']):
                company = metadata.get('company', 'Unknown')
                if company != user_company:
                    if company not in comparison_contexts:
                        comparison_contexts[company] = {'chunks': [], 'metadatas': []}
                    comparison_contexts[company]['chunks'].append(chunk)
                    comparison_contexts[company]['metadatas'].append(metadata)

        # Build comparative prompt
        prompt = self._build_comparative_prompt(
            question, user_company, user_context, comparison_contexts
        )

        # Generate answer
        print("💭 Generating comparative analysis...")
        answer = self._generate_answer(prompt)

        return {
            'question': question,
            'answer': answer,
            'user_company': user_company,
            'user_context': user_context,
            'comparison_contexts': comparison_contexts
        }

    def _get_company_context(
        self,
        question: str,
        company_name: str,
        top_k: int,
        use_hybrid: bool
    ) -> Dict[str, any]:
        """
        Get relevant context for a specific company
        """
        # Generate query embedding
        query_embedding = self.rag.embedder.encode(
            [question],
            device=self.rag.device
        )

        # Query ChromaDB for this specific company
        # First get all docs for this company
        all_docs = self.collection.get()

        # Filter by company
        company_chunks = []
        company_metadatas = []
        company_embeddings = []

        if all_docs and all_docs['metadatas']:
            for i, metadata in enumerate(all_docs['metadatas']):
                if metadata.get('company', '').lower() == company_name.lower():
                    company_chunks.append(all_docs['documents'][i])
                    company_metadatas.append(metadata)

        if not company_chunks:
            return {'chunks': [], 'metadatas': [], 'scores': []}

        # If using hybrid search and method exists
        if use_hybrid and hasattr(self.rag, 'hybrid_search'):
            # Use hybrid search with company filter
            results = self.rag.hybrid_search(question, top_k=top_k)

            # Filter results for this company
            filtered_chunks = []
            filtered_metadatas = []

            for chunk, metadata in zip(results['documents'][0], results['metadatas'][0]):
                if metadata.get('company', '').lower() == company_name.lower():
                    filtered_chunks.append(chunk)
                    filtered_metadatas.append(metadata)
                    if len(filtered_chunks) >= top_k:
                        break

            return {
                'chunks': filtered_chunks,
                'metadatas': filtered_metadatas,
                'scores': [1.0] * len(filtered_chunks)
            }
        else:
            # Use basic semantic search on company chunks
            # Create a temporary collection or use distance calculation
            from sklearn.metrics.pairwise import cosine_similarity

            # Get embeddings for company chunks
            chunk_embeddings = self.rag.embedder.encode(
                company_chunks[:100],  # Limit for performance
                device=self.rag.device
            )

            # Calculate similarities
            similarities = cosine_similarity(
                query_embedding.reshape(1, -1),
                chunk_embeddings
            )[0]

            # Get top-k
            top_indices = np.argsort(similarities)[::-1][:top_k]

            return {
                'chunks': [company_chunks[i] for i in top_indices],
                'metadatas': [company_metadatas[i] for i in top_indices],
                'scores': [float(similarities[i]) for i in top_indices]
            }

    def _get_general_context(
        self,
        question: str,
        top_k: int,
        use_hybrid: bool
    ) -> Dict[str, any]:
        """
        Get general context across all companies
        """
        if use_hybrid and hasattr(self.rag, 'hybrid_search'):
            results = self.rag.hybrid_search(question, top_k=top_k)
            return {
                'chunks': results['documents'][0] if results['documents'] else [],
                'metadatas': results['metadatas'][0] if results['metadatas'] else []
            }
        else:
            # Use basic query
            query_embedding = self.rag.embedder.encode([question], device=self.rag.device)
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k
            )
            return {
                'chunks': results['documents'][0] if results['documents'] else [],
                'metadatas': results['metadatas'][0] if results['metadatas'] else []
            }

    def _build_comparative_prompt(
        self,
        question: str,
        user_company: str,
        user_context: Dict,
        comparison_contexts: Dict[str, Dict]
    ) -> str:
        """
        Build a prompt for comparative analysis
        """
        prompt = f"""You are a financial analyst conducting comparative analysis of SEC filings.

Question: {question}

USER'S COMPANY: {user_company}
{'='*60}
"""

        # Add user company context
        if user_context['chunks']:
            prompt += f"\nRelevant information from {user_company}:\n"
            for i, (chunk, metadata) in enumerate(zip(user_context['chunks'][:5],
                                                       user_context['metadatas'][:5])):
                section = metadata.get('section', 'Unknown')
                year = metadata.get('year', 'Unknown')
                prompt += f"\n[{user_company} - {section} - {year}]\n{chunk}\n"
        else:
            prompt += f"\n⚠️ No relevant information found for {user_company}\n"

        # Add comparison companies context
        prompt += f"\n\nCOMPARISON COMPANIES:\n{'='*60}\n"

        for company, context in comparison_contexts.items():
            if context['chunks']:
                prompt += f"\n--- {company} ---\n"
                for i, (chunk, metadata) in enumerate(zip(context['chunks'][:3],
                                                          context['metadatas'][:3])):
                    section = metadata.get('section', 'Unknown')
                    year = metadata.get('year', 'Unknown')
                    prompt += f"\n[{company} - {section} - {year}]\n{chunk}\n"

        prompt += f"""

INSTRUCTIONS:
1. Analyze the information from {user_company} first
2. Compare with the information from other companies
3. Identify similarities and differences
4. Provide specific examples and metrics when available
5. Cite the company names and years in your answer
6. If information is missing for any company, state that clearly
7. Structure your answer with clear comparisons

Provide a comprehensive comparative analysis:"""

        return prompt

    def _generate_answer(self, prompt: str) -> str:
        """
        Generate answer using OpenAI API
        """
        try:
            import openai

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial analyst expert in comparative SEC filing analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )

            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {str(e)}\n\nPrompt was:\n{prompt}"

    def compare_metrics(
        self,
        user_company: str,
        comparison_companies: List[str],
        metric_type: str = "revenue"
    ) -> Dict[str, any]:
        """
        Compare specific financial metrics across companies

        Args:
            user_company: User's company name
            comparison_companies: List of companies to compare
            metric_type: Type of metric (revenue, profit, risk, etc.)

        Returns:
            Comparison results
        """
        metric_queries = {
            'revenue': 'What was the total revenue and revenue growth rate?',
            'profit': 'What was the net income and profit margin?',
            'risk': 'What are the main risk factors identified?',
            'assets': 'What are the total assets and key asset categories?',
            'debt': 'What is the total debt and debt-to-equity ratio?',
            'cashflow': 'What was the operating cash flow?'
        }

        query = metric_queries.get(
            metric_type.lower(),
            f"What are the key {metric_type} metrics?"
        )

        return self.ask_comparative_question(
            question=query,
            user_company=user_company,
            comparison_companies=comparison_companies,
            top_k=5
        )


# Example usage functions
def example_upload_and_compare(rag_instance):
    """
    Example workflow for uploading a filing and performing comparison
    """
    # Initialize managers
    uploader = UserFilingManager(rag_instance)
    analyzer = ComparativeAnalyzer(rag_instance)

    # Example 1: Upload a user's SEC filing
    print("="*60)
    print("STEP 1: Upload User's SEC Filing")
    print("="*60)

    upload_result = uploader.upload_user_filing(
        file_path="path/to/user_10k.pdf",  # User provides this
        company_name="My Company Inc",
        filing_type="10-K",
        fiscal_year="2023",
        section_name="Full Filing",
        cik="USER001"
    )

    print(f"\n✅ Upload Summary:")
    for key, value in upload_result.items():
        print(f"   {key}: {value}")

    # Example 2: List uploaded companies
    print("\n" + "="*60)
    print("STEP 2: View Uploaded Companies")
    print("="*60)

    uploaded = uploader.get_user_uploaded_companies()
    print(f"\nFound {len(uploaded)} user-uploaded companies:")
    for company in uploaded:
        print(f"   - {company['company_name']} ({company['cik']}) - {company['fiscal_year']}")

    # Example 3: Ask comparative question
    print("\n" + "="*60)
    print("STEP 3: Ask Comparative Question")
    print("="*60)

    result = analyzer.ask_comparative_question(
        question="How does the revenue growth compare across these companies?",
        user_company="My Company Inc",
        comparison_companies=["Apple Inc", "Microsoft Corporation"],
        top_k=5,
        use_hybrid=True
    )

    print(f"\nQuestion: {result['question']}")
    print(f"\nAnswer:\n{result['answer']}")

    # Example 4: Compare specific metrics
    print("\n" + "="*60)
    print("STEP 4: Compare Specific Metrics")
    print("="*60)

    metric_result = analyzer.compare_metrics(
        user_company="My Company Inc",
        comparison_companies=["Apple Inc", "Microsoft Corporation"],
        metric_type="risk"
    )

    print(f"\nMetric Comparison:\n{metric_result['answer']}")

    # Example 5: Delete uploaded filing (if needed)
    # uploader.delete_user_filing(cik="USER001")


if __name__ == "__main__":
    print("""
    User Filing Upload and Comparative Analysis Module
    ===================================================

    This module provides functionality to:
    1. Upload user SEC filings (PDF/TXT)
    2. Process and index them in ChromaDB
    3. Ask comparative questions
    4. Compare specific metrics

    See example_upload_and_compare() for usage examples.
    """)
