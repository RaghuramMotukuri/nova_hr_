"""
test_retrieval.py
Clean test suite for document retrieval:
  - Multi-tenant BM25 lexical search
  - Scoped company and subfolder search
"""
import os
import shutil
from pathlib import Path
import pytest
from src.embeddings import EmbeddingPipeline
from src.data_loader import load_all_documents

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
        pass

@pytest.fixture(scope="function", autouse=True)
def setup_test_environment():
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
    safe_rmtree(TEST_DATA_DIR)


def test_lexical_search_multi_tenant():
    pipeline = EmbeddingPipeline()
    docs = load_all_documents(str(TEST_DATA_DIR))
    pipeline.add_documents(docs)

    # Scoped Google search
    google_res = pipeline.lexical_search(query="annual leave", top_k=5, company="Google")
    assert len(google_res) > 0
    for res in google_res:
        assert res["company"] == "Google"
        assert "annual leave" in res["chunk_text"].lower()

    # Scoped Apple search
    apple_res = pipeline.lexical_search(query="sick leave", top_k=5, company="Apple")
    assert len(apple_res) > 0
    for res in apple_res:
        assert res["company"] == "Apple"
        assert "sick leave" in res["chunk_text"].lower()
