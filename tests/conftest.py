"""
conftest.py — Pytest configuration for LOVA_HR.
Isolates test runs from the production Firestore collection so synthetic test data
(e.g., Google/Apple test files) never pollutes the live 'hr_policies' collection.
"""
import os
import pytest

# Force tests to use an isolated test collection name in Firestore
os.environ["FIRESTORE_COLLECTION"] = "hr_policies_test"


@pytest.fixture(autouse=True)
def isolate_test_firestore_collection(monkeypatch):
    """Ensure every test runs against the isolated test collection."""
    monkeypatch.setenv("FIRESTORE_COLLECTION", "hr_policies_test")
