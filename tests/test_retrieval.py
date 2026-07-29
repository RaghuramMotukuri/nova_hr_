import os
import shutil
from pathlib import Path
import pytest
from src.embeddings import EmbeddingPipeline
from src.data_loader import load_all_documents

# Setup a temporary data directory for testing
TEST_DATA_DIR = Path("test_data")

def safe_rmtree(path: Path):
    if not path.exists():
        return
    for root, dirs, files in os.walk(str(path), topdown=False):
        for name in files:
            try:
                os.unlink(os.path.join(root, name))
            except Exception:
                pass
        for name in dirs:
            try:
                os.rmdir(os.path.join(root, name))
            except Exception:
                pass
    try:
        shutil.rmtree(path)
    except Exception:
        try:
            os.rmdir(path)
        except Exception:
            pass

@pytest.fixture(scope="function", autouse=True)
def setup_test_environment():
    # Setup
    safe_rmtree(TEST_DATA_DIR)
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create Company Google > Policies
    google_policies = TEST_DATA_DIR / "Google" / "Policies"
    google_policies.mkdir(parents=True, exist_ok=True)
    with open(google_policies / "annual_leave.txt", "w", encoding="utf-8") as f:
        f.write("Google Company policies on annual leave: Employees get 25 days of annual leave per calendar year. "
                "All annual leave requests must be approved by the manager in advance.")

    # Create Company Apple > Benefits
    apple_benefits = TEST_DATA_DIR / "Apple" / "Benefits"
    apple_benefits.mkdir(parents=True, exist_ok=True)
    with open(apple_benefits / "medical.txt", "w", encoding="utf-8") as f:
        f.write("Apple Health Benefits: Dental coverage and vision care are 100% sponsored. "
                "Medical insurance is fully covered for employees and their dependents.")

    # Create Company Apple > Policies
    apple_policies = TEST_DATA_DIR / "Apple" / "Policies"
    apple_policies.mkdir(parents=True, exist_ok=True)
    with open(apple_policies / "sick_leave.txt", "w", encoding="utf-8") as f:
        f.write("Apple Sick Leave Policy: Employees get 12 days of sick leave annually. "
                "Medical certificate is required for leaves extending beyond 3 consecutive days.")

    yield
    
    # Teardown
    safe_rmtree(TEST_DATA_DIR)


def test_data_loading():
    docs = load_all_documents(str(TEST_DATA_DIR))
    assert len(docs) >= 3
    
    # Check Google doc metadata
    google_docs = [doc for doc in docs if doc.metadata.get("company") == "Google"]
    assert len(google_docs) > 0
    assert google_docs[0].metadata["subfolder"] == "Policies"
    assert google_docs[0].metadata["filename"] == "annual_leave.txt"

    # Check Apple doc metadata
    apple_docs = [doc for doc in docs if doc.metadata.get("company") == "Apple"]
    assert len(apple_docs) >= 2
    subfolders = [doc.metadata["subfolder"] for doc in apple_docs]
    assert "Benefits" in subfolders
    assert "Policies" in subfolders


def test_scoped_lexical_and_semantic_searching():
    pipeline = EmbeddingPipeline(
        chunk_size=100, 
        chunk_overlap=20
    )
    pipeline.vector_store.clear()
    
    docs = load_all_documents(str(TEST_DATA_DIR))
    chunks = pipeline.chunk_documents(docs)
    
    # Index
    pipeline.sync_vector_store(chunks)
    
    # Test Scoped Lexical Search (within Google only)
    # Search for "leave" (both Google and Apple have leave documents)
    google_lex = pipeline.lexical_search(
        query="leave",
        chunks=chunks,
        top_k=2,
        company="Google"
    )
    assert len(google_lex) > 0
    for res in google_lex:
        assert res["company"] == "Google"
        assert "annual leave" in res["chunk_text"].lower()

    # Test Scoped Lexical Search (within Apple only)
    apple_lex = pipeline.lexical_search(
        query="leave",
        chunks=chunks,
        top_k=2,
        company="Apple"
    )
    assert len(apple_lex) > 0
    for res in apple_lex:
        assert res["company"] == "Apple"
        assert "sick leave" in res["chunk_text"].lower()

    # Test Scoped Semantic Search (within Apple > Benefits only)
    apple_sem_benefits = pipeline.semantic_search(
        query="insurance coverage",
        top_k=2,
        company="Apple",
        subfolder="Benefits"
    )
    assert len(apple_sem_benefits) > 0
    for res in apple_sem_benefits:
        assert res["company"] == "Apple"
        assert res["subfolder"] == "Benefits"
        assert "insurance" in res["chunk_text"].lower() or "dental" in res["chunk_text"].lower()


def test_scoped_chunk_deletion():
    pipeline = EmbeddingPipeline(
        chunk_size=100, 
        chunk_overlap=20
    )
    pipeline.vector_store.clear()
    
    docs = load_all_documents(str(TEST_DATA_DIR))
    chunks = pipeline.chunk_documents(docs)
    pipeline.sync_vector_store(chunks)
    
    # Delete Google file
    target_file = TEST_DATA_DIR / "Google" / "Policies" / "annual_leave.txt"
    pipeline.vector_store.delete_file_chunks(str(target_file.resolve()))
    
    # Search Google again, should return no semantic results
    google_res = pipeline.semantic_search(
        query="annual leave",
        top_k=2,
        company="Google"
    )
    assert len(google_res) == 0


def test_delete_company_and_subfolder_chunks():
    pipeline = EmbeddingPipeline(chunk_size=100, chunk_overlap=20)
    pipeline.vector_store.clear()
    
    docs = load_all_documents(str(TEST_DATA_DIR))
    chunks = pipeline.chunk_documents(docs)
    pipeline.sync_vector_store(chunks)
    
    # Delete Apple Benefits subfolder
    pipeline.vector_store.delete_subfolder_chunks("Apple", "Benefits")
    benefits_res = pipeline.semantic_search(
        query="insurance dental medical",
        top_k=2,
        company="Apple",
        subfolder="Benefits"
    )
    assert len(benefits_res) == 0
    
    # Apple Policies should still exist
    policies_res = pipeline.semantic_search(
        query="sick leave",
        top_k=2,
        company="Apple",
        subfolder="Policies"
    )
    assert len(policies_res) > 0

    # Delete entire Apple company
    pipeline.vector_store.delete_company_chunks("Apple")
    apple_res = pipeline.semantic_search(
        query="leave medical insurance",
        top_k=5,
        company="Apple"
    )
    assert len(apple_res) == 0
