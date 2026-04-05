"""
data/ingest_sec_filings.py

Real SEC EDGAR data ingestion pipeline.
Downloads 10-K and 10-Q filings from major public companies.

Run with: python -m data.ingest_sec_filings
"""

import os
import time
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.retriever import ingest_documents

load_dotenv()

COMPANIES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.",
    "CRM": "Salesforce Inc.",
    "NFLX": "Netflix Inc.",
    "TSLA": "Tesla Inc.",
    "AMZN": "Amazon.com Inc.",
    "META": "Meta Platforms Inc.",
}

FILING_TYPES = ["10-K", "10-Q"]
MAX_FILINGS_PER_TYPE = 2

TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def ingest_company_filings(ticker: str, company_name: str) -> int:
    from edgar import Company
    print(f"\n📥 Processing {company_name} ({ticker})...")
    total_chunks = 0

    try:
        company = Company(ticker)

        for filing_type in FILING_TYPES:
            try:
                entity_filings = company.get_filings(form=filing_type)
                if not entity_filings:
                    print(f"   ⚠️  No {filing_type} filings found")
                    continue

                count = min(MAX_FILINGS_PER_TYPE, len(entity_filings))
                for i in range(count):
                    filing = entity_filings[i]
                    try:
                        filing_date = str(getattr(filing, 'filing_date', 'unknown'))
                        print(f"   📄 Processing {filing_type} ({filing_date})...")

                        text = str(filing.text())[:50000]
                        if not text or len(text) < 100:
                            print(f"   ⚠️  Insufficient text, skipping")
                            continue

                        chunks = TEXT_SPLITTER.split_text(text)
                        documents = [
                            Document(
                                page_content=chunk,
                                metadata={
                                    "source": f"{ticker}_{filing_type}_{filing_date}",
                                    "ticker": ticker,
                                    "company": company_name,
                                    "filing_type": filing_type,
                                    "filing_date": filing_date,
                                    "chunk_index": j,
                                }
                            )
                            for j, chunk in enumerate(chunks)
                        ]

                        ingest_documents(documents)
                        total_chunks += len(documents)
                        print(f"   ✅ Ingested {len(documents)} chunks")
                        time.sleep(0.5)

                    except Exception as e:
                        print(f"   ❌ Filing error: {e}")
                        continue

            except Exception as e:
                print(f"   ❌ {filing_type} error: {e}")
                continue

    except Exception as e:
        print(f"❌ Company error for {ticker}: {e}")

    return total_chunks


def run_ingestion():
    from edgar import set_identity
    set_identity("Akash Chaudhari akashchaudhariofficial44@gmail.com")

    # Check if already ingested
    vs = get_vector_store()
    try:
        existing = vs.get()
        existing_sources = set(
            m.get("source", "") for m in existing.get("metadatas", [])
        )
        print(f"📦 Vector store already has {len(existing_sources)} unique sources")
    except Exception:
        existing_sources = set()

    # Skip companies already ingested
    def should_skip(ticker: str, filing_type: str, filing_date: str) -> bool:
        source_key = f"{ticker}_{filing_type}_{filing_date}"
        return source_key in existing_sources


if __name__ == "__main__":
    run_ingestion()