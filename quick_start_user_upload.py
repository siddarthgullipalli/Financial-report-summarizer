"""
QUICK START: Upload Your SEC Filing and Compare with Other Companies

This is a simple, ready-to-use script. Just update the file paths and run!
"""

# ============================================================================
# STEP 1: Import Required Libraries
# ============================================================================
from user_filing_upload import UserFilingManager, ComparativeAnalyzer

# Assuming you have 'rag' instance from GENAI_PROJECT_CHROMADB.ipynb
# If not, uncomment and run the initialization code below:

"""
from sentence_transformers import SentenceTransformer
import chromadb
import torch
import os

class FinBERTFinancialRAG:
    def __init__(self, persist_directory="~/FinancialAI/chromadb"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Load FinBERT embedder
        self.embedder = SentenceTransformer('yiyanghkust/finbert-tone')
        self.embedder.to(self.device)

        # Initialize ChromaDB
        persist_directory = os.path.expanduser(persist_directory)
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)

        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(name="financial_filings")
            print(f"Loaded existing collection")
        except:
            self.collection = self.chroma_client.create_collection(
                name="financial_filings",
                metadata={"description": "Financial SEC filings"}
            )
            print("Created new collection")

# Create RAG instance
rag = FinBERTFinancialRAG()
"""

# ============================================================================
# STEP 2: Initialize Managers
# ============================================================================
uploader = UserFilingManager(rag)
analyzer = ComparativeAnalyzer(rag)

print("✅ Managers initialized!\n")

# ============================================================================
# STEP 3: Upload Your SEC Filing
# ============================================================================
print("="*70)
print("UPLOADING YOUR SEC FILING")
print("="*70)

# 🔥 CHANGE THESE VALUES TO YOUR ACTUAL FILE AND COMPANY INFO 🔥
YOUR_FILE_PATH = "/path/to/your/sec_filing.pdf"  # ⬅️ UPDATE THIS
YOUR_COMPANY_NAME = "My Startup Inc"              # ⬅️ UPDATE THIS
YOUR_FILING_TYPE = "10-K"                         # 10-K, 10-Q, etc.
YOUR_FISCAL_YEAR = "2023"                         # Year
YOUR_CIK = "USER_MYSTARTUP"                       # Custom identifier

upload_result = uploader.upload_user_filing(
    file_path=YOUR_FILE_PATH,
    company_name=YOUR_COMPANY_NAME,
    filing_type=YOUR_FILING_TYPE,
    fiscal_year=YOUR_FISCAL_YEAR,
    section_name="Full Filing",
    cik=YOUR_CIK
)

print(f"\n✅ Successfully uploaded {YOUR_COMPANY_NAME}!")
print(f"   Chunks created: {upload_result['chunks_created']}")
print(f"   Total characters: {upload_result['total_characters']}")

# ============================================================================
# STEP 4: Compare with Other Companies
# ============================================================================
print("\n" + "="*70)
print("COMPARATIVE ANALYSIS")
print("="*70)

# 🔥 CHANGE THESE TO THE COMPANIES YOU WANT TO COMPARE WITH 🔥
COMPANIES_TO_COMPARE = [
    "APPLE INC",
    "MICROSOFT CORP",
    "ALPHABET INC"
]

# Example 1: Revenue Comparison
print("\n📊 Question 1: Revenue Growth Comparison")
print("-"*70)

result1 = analyzer.ask_comparative_question(
    question="How does the revenue growth rate compare? What are the main revenue sources?",
    user_company=YOUR_COMPANY_NAME,
    comparison_companies=COMPANIES_TO_COMPARE,
    top_k=5
)

print(f"\n{result1['answer']}\n")

# Example 2: Risk Factors
print("\n⚠️  Question 2: Risk Factors Comparison")
print("-"*70)

result2 = analyzer.ask_comparative_question(
    question="What are the main risk factors identified by each company?",
    user_company=YOUR_COMPANY_NAME,
    comparison_companies=COMPANIES_TO_COMPARE,
    top_k=5
)

print(f"\n{result2['answer']}\n")

# Example 3: Business Strategy
print("\n🎯 Question 3: Business Strategy Comparison")
print("-"*70)

result3 = analyzer.ask_comparative_question(
    question="What are the key competitive advantages and business strategies?",
    user_company=YOUR_COMPANY_NAME,
    comparison_companies=COMPANIES_TO_COMPARE,
    top_k=5
)

print(f"\n{result3['answer']}\n")

# ============================================================================
# STEP 5: Metric-Specific Comparisons
# ============================================================================
print("\n" + "="*70)
print("METRIC-SPECIFIC COMPARISONS")
print("="*70)

# Revenue metrics
print("\n💰 Revenue Metrics:")
print("-"*70)
revenue_result = analyzer.compare_metrics(
    user_company=YOUR_COMPANY_NAME,
    comparison_companies=COMPANIES_TO_COMPARE[:2],  # Compare with top 2
    metric_type="revenue"
)
print(revenue_result['answer'])

# Profit metrics
print("\n📈 Profit Metrics:")
print("-"*70)
profit_result = analyzer.compare_metrics(
    user_company=YOUR_COMPANY_NAME,
    comparison_companies=COMPANIES_TO_COMPARE[:2],
    metric_type="profit"
)
print(profit_result['answer'])

# ============================================================================
# STEP 6: View All Uploaded Companies
# ============================================================================
print("\n" + "="*70)
print("ALL USER-UPLOADED COMPANIES")
print("="*70)

uploaded_companies = uploader.get_user_uploaded_companies()
for i, company in enumerate(uploaded_companies, 1):
    print(f"\n{i}. {company['company_name']}")
    print(f"   CIK: {company['cik']}")
    print(f"   Filing: {company['filing_type']} ({company['fiscal_year']})")
    print(f"   Uploaded: {company['upload_timestamp']}")

# ============================================================================
# OPTIONAL: Custom Questions
# ============================================================================
print("\n" + "="*70)
print("CUSTOM QUESTIONS (ADD YOUR OWN!)")
print("="*70)

# 🔥 ADD YOUR OWN QUESTIONS HERE 🔥
custom_questions = [
    "How do the companies describe their market position?",
    "What technology investments are mentioned?",
    "How do they approach sustainability and ESG?",
]

for i, question in enumerate(custom_questions, 1):
    print(f"\n❓ Question {i}: {question}")
    print("-"*70)

    result = analyzer.ask_comparative_question(
        question=question,
        user_company=YOUR_COMPANY_NAME,
        comparison_companies=COMPANIES_TO_COMPARE,
        top_k=3
    )

    print(f"\n{result['answer']}\n")

print("\n" + "="*70)
print("✅ ANALYSIS COMPLETE!")
print("="*70)
