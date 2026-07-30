"""
config.py
Firebase Admin SDK initialization for LOVA_HR.

Credential resolution order:
  1. FIREBASE_SERVICE_ACCOUNT_KEY env var  (JSON string — good for CI/CD)
  2. serviceAccountKey.json in the repo root (good for local dev)
  3. Application Default Credentials (ADC) via GOOGLE_APPLICATION_CREDENTIALS
     or `gcloud auth application-default login`

Usage:
    from src.config import init_firebase, get_firestore_client, COLLECTION_NAME
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import firebase_admin
from firebase_admin import credentials, firestore

# ── Constants ────────────────────────────────────────────────────────────────

# Firebase Project ID
FIREBASE_PROJECT_ID: str = os.getenv("FIREBASE_PROJECT_ID", "lova-hr")

# Firestore collection that stores all HR policy chunks
COLLECTION_NAME: str = os.getenv("FIRESTORE_COLLECTION", "hr_policies")

# BGE model identifiers (overridable via env)
BGE_EMBEDDING_MODEL: str = os.getenv("BGE_EMBEDDING_MODEL", "BAAI/bge-m3")
BGE_RERANKER_MODEL: str = os.getenv("BGE_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# Embedding dimensionality for BAAI/bge-m3
EMBEDDING_DIM: int = 1024

# Path to the local service-account key file (repo root)
_DEFAULT_KEY_FILE = Path(__file__).resolve().parent.parent / "serviceAccountKey.json"

# Firebase app name (allows multiple initializations in tests)
_APP_NAME = "lova_hr"

# Module-level cache
_firebase_app: Optional[firebase_admin.App] = None
_firestore_client = None


# ── Initialization ────────────────────────────────────────────────────────────

def init_firebase() -> firebase_admin.App:
    """
    Initialize the Firebase Admin SDK and return the App instance.
    Idempotent — safe to call multiple times.
    """
    global _firebase_app

    # Already initialized
    if _firebase_app is not None:
        return _firebase_app

    # Check if the named app already exists (e.g. from a previous Streamlit hot-reload)
    try:
        _firebase_app = firebase_admin.get_app(_APP_NAME)
        return _firebase_app
    except ValueError:
        pass  # App not yet initialized

    cred = _resolve_credentials()

    _firebase_app = firebase_admin.initialize_app(cred, name=_APP_NAME)
    print(f"[Firebase] Initialized app '{_APP_NAME}' successfully.")
    return _firebase_app


def _resolve_credentials() -> credentials.Base:
    """
    Resolve Firebase credentials from environment or local key file.
    Raises EnvironmentError with actionable instructions if nothing is found.
    """
    # 1. JSON string from environment variable
    key_json_str = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY", "").strip()
    if key_json_str:
        try:
            key_dict = json.loads(key_json_str)
            print("[Firebase] Using credentials from FIREBASE_SERVICE_ACCOUNT_KEY env var.")
            return credentials.Certificate(key_dict)
        except json.JSONDecodeError as exc:
            raise EnvironmentError(
                "FIREBASE_SERVICE_ACCOUNT_KEY is set but contains invalid JSON.\n"
                f"Parse error: {exc}"
            ) from exc

    # 2. Local serviceAccountKey.json or any *-firebase-adminsdk-*.json file in repo root
    repo_root = Path(__file__).resolve().parent.parent
    key_files = list(repo_root.glob("serviceAccountKey*.json")) + list(repo_root.glob("*firebase-adminsdk*.json"))
    if not key_files:
        for json_file in repo_root.glob("*.json"):
            if json_file.name in ["pyrightconfig.json", "package.json", "tsconfig.json"]:
                continue
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    if isinstance(content, dict) and content.get("type") == "service_account":
                        key_files.append(json_file)
                        break
            except Exception:
                pass

    if key_files:
        key_path = key_files[0]
        print(f"[Firebase] Using credentials from '{key_path.name}'.")
        return credentials.Certificate(str(key_path))

    # 3. Application Default Credentials (ADC)
    adc_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    if adc_path and Path(adc_path).exists():
        print(f"[Firebase] Using Application Default Credentials from '{adc_path}'.")
        return credentials.Certificate(adc_path)

    # Nothing found — raise a helpful error
    raise EnvironmentError(
        "\n"
        "═══════════════════════════════════════════════════════════════\n"
        "  Firebase credentials not found. Choose ONE of:\n"
        "\n"
        "  Option A — Local file (recommended for dev):\n"
        "    1. Go to Firebase Console → Project Settings → Service Accounts\n"
        "    2. Click 'Generate new private key' → save as serviceAccountKey.json\n"
        "    3. Place serviceAccountKey.json in the repository root.\n"
        "\n"
        "  Option B — Environment variable (recommended for CI/CD):\n"
        "    export FIREBASE_SERVICE_ACCOUNT_KEY='<contents of serviceAccountKey.json>'\n"
        "\n"
        "  Option C — Application Default Credentials:\n"
        "    gcloud auth application-default login\n"
        "═══════════════════════════════════════════════════════════════\n"
    )


def get_firestore_client():
    """
    Return an authenticated Firestore client.
    Initializes Firebase if not already done.
    """
    global _firestore_client
    if _firestore_client is not None:
        return _firestore_client

    init_firebase()
    # Pass the named app so we always use the correct project
    _firestore_client = firestore.client(app=firebase_admin.get_app(_APP_NAME))
    return _firestore_client


def is_firebase_available() -> bool:
    """
    Non-throwing probe: returns True if Firebase can be initialized, False otherwise.
    Used by the app to decide whether to show an offline/fallback warning.
    """
    try:
        init_firebase()
        return True
    except Exception:
        return False
