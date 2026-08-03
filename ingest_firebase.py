#!/usr/bin/env python3
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
"""
ingest_firebase.py
CLI script to parse local policy files, chunk them, compute BGE embeddings
(BAAI/bge-large-en-v1.5), and upload them to Google Cloud Firestore.

Usage:
    python ingest_firebase.py [OPTIONS]

Options:
    --data-dir PATH     Directory containing policy documents (default: data)
    --clear             Wipe the Firestore collection before ingesting
    --batch-size INT    Embedding batch size (default: 50)
    --dry-run           Parse & chunk without uploading to Firestore
    --help              Show this message and exit

Examples:
    # Full ingest from data/ directory
    python ingest_firebase.py

    # Clear existing data and re-ingest
    python ingest_firebase.py --clear

    # Ingest from a custom directory with smaller batches (low-memory machines)
    python ingest_firebase.py --data-dir /path/to/policies --batch-size 20

    # Test chunking without touching Firestore
    python ingest_firebase.py --dry-run

Firestore Vector Index (run ONCE before first query):
    gcloud firestore indexes composite create \\
      --project=YOUR_PROJECT_ID \\
      --collection-group=hr_policies \\
      --query-scope=COLLECTION \\
      --field-config field-path=embedding,vector-config='{"dimension":"1024","flat":{}}'

    Check index status:
    gcloud firestore indexes composite list --project=YOUR_PROJECT_ID
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ── Allow running from repo root without installing as a package ──────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest HR policy documents into Google Cloud Firestore.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing policy documents (default: data)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete all existing Firestore documents before ingesting",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Embedding batch size (default: 50). Reduce on low-memory machines.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk documents but do NOT upload to Firestore",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Text chunk size in characters (default: 800)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Chunk overlap in characters (default: 150)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()

    print("=" * 70)
    print("  LOVA_HR — Firebase Ingestion Script")
    print("=" * 70)
    print(f"  Data directory : {data_dir}")
    print(f"  Clear first    : {args.clear}")
    print(f"  Batch size     : {args.batch_size}")
    print(f"  Dry run        : {args.dry_run}")
    print(f"  Chunk size     : {args.chunk_size} chars")
    print(f"  Chunk overlap  : {args.chunk_overlap} chars")
    print("=" * 70)

    if not data_dir.exists():
        print(f"\n❌ Data directory not found: {data_dir}")
        sys.exit(1)

    # ── Load environment variables ──────────────────────────────────────────
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("[Ingest] .env loaded.")
    except ImportError:
        pass

    # ── Load documents ──────────────────────────────────────────────────────
    print("\n[Ingest] 📄 Loading documents...")
    t0 = time.time()
    from src.data_loader import load_all_documents
    documents = load_all_documents(str(data_dir))

    if not documents:
        print(f"\n⚠️  No documents found in '{data_dir}'. Exiting.")
        sys.exit(0)

    print(f"[Ingest] Loaded {len(documents)} pages/documents in {time.time() - t0:.1f}s.")

    # ── Chunk documents ─────────────────────────────────────────────────────
    print("\n[Ingest] ✂️  Chunking documents...")
    t1 = time.time()
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from src.preprocessor import sanitize_chunk, is_junk_chunk

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    raw_chunks = splitter.split_documents(documents)
    chunks = []
    for c in raw_chunks:
        clean = sanitize_chunk(c.page_content)
        if clean and not is_junk_chunk(clean):
            c.page_content = clean
            chunks.append(c)
    print(f"[Ingest] {len(documents)} pages → {len(chunks)} sanitized chunks in {time.time() - t1:.1f}s.")

    if args.dry_run:
        print("\n🏁 Dry run complete. No data was written to Firestore.")
        print(f"   Would upload {len(chunks)} chunks.")
        # Print a sample chunk
        if chunks:
            sample = chunks[0]
            print(f"\nSample chunk metadata: {sample.metadata}")
            print(f"Sample chunk text (first 300 chars):\n{sample.page_content[:300]}...")
        sys.exit(0)

    # ── Initialize Firebase ─────────────────────────────────────────────────
    print("\n[Ingest] 🔥 Initializing Firebase...")
    try:
        from src.config import init_firebase, COLLECTION_NAME
        init_firebase()
        print(f"[Ingest] Firebase ready. Target collection: '{COLLECTION_NAME}'")
    except EnvironmentError as exc:
        print(f"\n❌ Firebase credential error:\n{exc}")
        print("👉 Tip: Run with --dry-run to test document loading without Firebase.")
        sys.exit(1)
    except Exception as exc:
        print(f"\n❌ Firebase initialization failed: {exc}")
        sys.exit(1)

    # ── Upload to Firestore ─────────────────────────────────────────────────
    print("\n[Ingest] ⬆️  Uploading to Firestore...")
    t2 = time.time()
    from src.firebase_client import FirestoreVectorStore
    store = FirestoreVectorStore(batch_size=args.batch_size)

    store.upload_policy_chunks(chunks, force_reload=args.clear)

    elapsed = time.time() - t2
    print(f"\n[Ingest] ✅ Upload complete in {elapsed:.1f}s.")

    # ── Print final stats ───────────────────────────────────────────────────
    try:
        total = store.count()
        print(f"[Ingest] 📊 Total documents in Firestore collection: {total}")
    except Exception:
        pass

    print("\n" + "=" * 70)
    print("  Next steps:")
    print("  1. Create the Firestore vector index (if not done already):")
    print()
    print("     gcloud firestore indexes composite create \\")
    print("       --project=YOUR_PROJECT_ID \\")
    print("       --collection-group=hr_policies \\")
    print("       --query-scope=COLLECTION \\")
    print("       --field-config field-path=embedding,vector-config='{\"dimension\":\"1024\",\"flat\":{}}'")
    print()
    print("  2. Check index status:")
    print("     gcloud firestore indexes composite list --project=YOUR_PROJECT_ID")
    print()
    print("  3. Start the app:")
    print("     streamlit run app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
