"""
generator.py
Multi-Output LLM Answer Generator for LOVA_HR.

Collects answers from ALL configured providers (up to max_providers) and
returns every successful response so the UI can display them as separate cards.

Providers (run concurrently where possible):
  1. Hugging Face Serverless API (HF_TOKEN)  → microsoft/Phi-3.5-mini-instruct
  2. Groq Cloud API           (GROQ_API_KEY) → llama-3.1-8b-instant
  3. Grounded Extractive Synthesis           → always produced as a baseline card
"""
import os
import concurrent.futures
from typing import Any, Dict, List, Optional
import requests


# ── Provider metadata ─────────────────────────────────────────────────────────
_PROVIDERS = {
    "phi35": {
        "label": "Phi-3.5-mini-instruct",
        "icon": "🧠",
        "badge_color": "#38bdf8",        # sky-blue
        "badge_bg": "rgba(56,189,248,0.12)",
        "border": "#38bdf8",
    },
    "groq": {
        "label": "llama-3.1-8b-instant",
        "icon": "⚡",
        "badge_color": "#a78bfa",        # violet
        "badge_bg": "rgba(167,139,250,0.12)",
        "border": "#a78bfa",
    },
    "extractive": {
        "label": "Extractive Grounded Fallback",
        "icon": "📋",
        "badge_color": "#34d399",        # emerald
        "badge_bg": "rgba(52,211,153,0.10)",
        "border": "#34d399",
    },
}


class LLMGenerator:
    """
    Multi-Output Free-Tier LLM Answer Generator for LOVA_HR.

    generate_answer() returns:
    {
        "answers":   [{"provider_key": str, "answer": str}, ...],  # one per provider
        "citations": [...],                                          # shared source list
    }
    Each item in `answers` has a `provider_key` that maps to _PROVIDERS above.
    """

    def __init__(self):
        self.groq_api_key: str = os.getenv("GROQ_API_KEY", "").strip()
        self.hf_token: str = os.getenv("HF_TOKEN", "").strip()

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        company: Optional[str] = None,
        max_providers: int = 3,
    ) -> Dict[str, Any]:
        """
        Run configured providers concurrently and return up to `max_providers`
        successful answers.

        max_providers controls the slider in the sidebar:
          1 → Phi-3.5 only (or best available)
          2 → Phi-3.5 + Groq
          3 → Phi-3.5 + Groq + Extractive baseline

        Returns:
            {
                "answers":   [{"provider_key": str, "answer": str}, ...],
                "citations": [...],
                # legacy flat keys for backward-compat:
                "answer":    str,
                "provider":  str,
            }
        """
        if not chunks:
            empty = "No relevant policy documents were found in scope to answer this question."
            return {
                "answers": [{"provider_key": "extractive", "answer": empty}],
                "answer": empty,
                "citations": [],
                "provider": "none",
            }

        # Build shared citation list and context string
        context_blocks: List[str] = []
        citations: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(chunks, 1):
            c_name   = chunk.get("company",   chunk.get("company_id",   "General"))
            sub_name = chunk.get("subfolder", "General")
            f_name   = chunk.get("filename",  chunk.get("document_id",  "doc"))
            page_num = chunk.get("page_number", chunk.get("page", 1))
            cite_tag = f"[{c_name} > {sub_name} > {f_name} (p. {page_num})]"
            context_blocks.append(f"Source #{idx} {cite_tag}:\n{chunk['chunk_text']}")
            citations.append({
                "rank":        idx,
                "company":     c_name,
                "subfolder":   sub_name,
                "filename":    f_name,
                "page":        page_num,
                "source_file": chunk.get("source_file", ""),
                "score":       chunk.get("reranker_score", chunk.get("score", 0.0)),
            })

        formatted_context = "\n\n".join(context_blocks)
        target_scope = company if (company and company != "All Companies") else "the selected organization"

        # Extractive baseline (always included, built synchronously — zero latency)
        top_chunk = chunks[0]
        top_cite  = (
            f"{top_chunk.get('company')} > "
            f"{top_chunk.get('subfolder')} > "
            f"{top_chunk.get('filename')}"
        )
        extractive_text = (
            f"**Direct policy excerpt from {top_cite}:**\n\n"
            f"> {top_chunk['chunk_text']}"
        )

        # Collect all answers (API calls run in a thread pool)
        tasks: Dict[str, Any] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            if self.hf_token:
                tasks["phi35"] = pool.submit(
                    self._call_phi, query, formatted_context, target_scope
                )
            if self.groq_api_key:
                tasks["groq"] = pool.submit(
                    self._call_groq, query, formatted_context, target_scope
                )

        answers: List[Dict[str, Any]] = []

        # Phi-3.5 result
        if "phi35" in tasks:
            try:
                answers.append({"provider_key": "phi35", "answer": tasks["phi35"].result()})
            except Exception as exc:
                print(f"[LLMGenerator] Phi-3.5 error: {exc}")

        # Groq result
        if "groq" in tasks:
            try:
                answers.append({"provider_key": "groq", "answer": tasks["groq"].result()})
            except Exception as exc:
                print(f"[LLMGenerator] Groq error: {exc}")

        # Always add the extractive baseline last
        answers.append({"provider_key": "extractive", "answer": extractive_text})

        # Respect max_providers: trim to the requested count
        answers = answers[:max(1, max_providers)]

        # Legacy flat fields
        best = answers[0] if answers else {"provider_key": "extractive", "answer": extractive_text}
        provider_meta = _PROVIDERS.get(best["provider_key"], {})

        return {
            "answers":   answers,
            "citations": citations,
            # legacy compat
            "answer":   best["answer"],
            "provider": provider_meta.get("label", best["provider_key"]),
        }

    # ── Private call helpers ──────────────────────────────────────────────────

    def _call_phi(self, query: str, context: str, scope: str) -> str:
        """Call microsoft/Phi-3.5-mini-instruct via the HuggingFace Inference API."""
        url = (
            "https://api-inference.huggingface.co/models/"
            "microsoft/Phi-3.5-mini-instruct/v1/chat/completions"
        )
        system_prompt = (
            f"You are LOVA_HR, an expert HR policy assistant for {scope}. "
            "Answer questions concisely and factually using ONLY the provided policy context chunks. "
            "For every claim, cite the source using the [Source #N] tag shown in the context. "
            "If the answer is not in the context, say: 'The policy documents in scope do not specify this.' "
            "Do NOT invent or guess information beyond what is given."
        )
        payload = {
            "model": "microsoft/Phi-3.5-mini-instruct",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"POLICY CONTEXT:\n{context}\n\nUSER QUESTION: {query}"},
            ],
            "max_tokens": 512,
            "temperature": 0.1,
        }
        res = requests.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {self.hf_token}", "Content-Type": "application/json"},
            timeout=30,
        )
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"].strip()

    def _call_groq(self, query: str, context: str, scope: str) -> str:
        """Call llama-3.1-8b-instant via the Groq Cloud API."""
        system_prompt = (
            "You are LOVA_HR, an expert multi-tenant HR policy chatbot. "
            "Answer user questions concisely and factually, strictly using the provided context chunks for "
            f"{scope}. Do NOT use outside information or guess. If the answer is not in the context, "
            "say 'The policy document in scope does not specify this information.' "
            "Always mention source document names."
        )
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"POLICY CONTEXT:\n{context}\n\nUSER QUESTION: {query}"},
            ],
            "temperature": 0.1,
            "max_tokens":  512,
        }
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.groq_api_key}", "Content-Type": "application/json"},
            timeout=10,
        )
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"].strip()

    # ── Legacy compat (kept so old call sites still work) ─────────────────────
    def _call_huggingface(self, query: str, context: str, scope: str) -> str:
        return self._call_phi(query, context, scope)
