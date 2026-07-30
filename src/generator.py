"""
generator.py
Free-Tier LLM Answer Generator for LOVA_HR.

Provides zero-VRAM, zero-cost conversational answer generation with grounded policy citations.
Order of execution:
  1. Groq Cloud API (GROQ_API_KEY) → 'llama-3.1-8b-instant'
  2. Hugging Face Serverless API (HF_TOKEN) → 'Qwen/Qwen2.5-7B-Instruct'
  3. Grounded Extractive Synthesis (Fallback when no API key is configured)
"""
import os
from typing import Any, Dict, List, Optional
import requests


class LLMGenerator:
    """
    Zero-VRAM Free-Tier LLM Answer Generator for LOVA_HR.
    """

    def __init__(self):
        self.groq_api_key: str = os.getenv("GROQ_API_KEY", "").strip()
        self.hf_token: str = os.getenv("HF_TOKEN", "").strip()

    def generate_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        company: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a factual answer derived strictly from context chunks with citations.
        """
        if not chunks:
            return {
                "answer": "No relevant policy documents were found in scope to answer this question.",
                "citations": [],
                "provider": "none",
            }

        context_blocks = []
        citations = []
        for idx, chunk in enumerate(chunks, 1):
            c_name = chunk.get("company", chunk.get("company_id", "General"))
            sub_name = chunk.get("subfolder", "General")
            f_name = chunk.get("filename", chunk.get("document_id", "doc"))
            page_num = chunk.get("page_number", chunk.get("page", 1))
            cite_tag = f"[{c_name} > {sub_name} > {f_name} (p. {page_num})]"

            context_blocks.append(f"Source #{idx} {cite_tag}:\n{chunk['chunk_text']}")
            citations.append({
                "rank": idx,
                "company": c_name,
                "subfolder": sub_name,
                "filename": f_name,
                "page": page_num,
                "source_file": chunk.get("source_file", ""),
                "score": chunk.get("reranker_score", chunk.get("score", 0.0)),
            })

        formatted_context = "\n\n".join(context_blocks)
        target_scope = company if (company and company != "All Companies") else "the selected organization"

        # 1. Groq Cloud API (llama-3.1-8b-instant)
        if self.groq_api_key:
            try:
                ans = self._call_groq(query, formatted_context, target_scope)
                return {
                    "answer": ans,
                    "citations": citations,
                    "provider": "Groq (llama-3.1-8b-instant)",
                }
            except Exception as exc:
                print(f"[LLMGenerator] Groq API notice: {exc}")

        # 2. Hugging Face Serverless API (Qwen2.5-7B-Instruct)
        if self.hf_token:
            try:
                ans = self._call_huggingface(query, formatted_context, target_scope)
                return {
                    "answer": ans,
                    "citations": citations,
                    "provider": "Hugging Face (Qwen2.5-7B-Instruct)",
                }
            except Exception as exc:
                print(f"[LLMGenerator] Hugging Face API notice: {exc}")

        # 3. Grounded Extractive Fallback
        top_chunk = chunks[0]
        top_cite = f"{top_chunk.get('company')} > {top_chunk.get('subfolder')} > {top_chunk.get('filename')}"
        extracted_answer = (
            f"Based on **{top_cite}**:\n\n"
            f"> {top_chunk['chunk_text']}\n\n"
            f"*Tip: Set `GROQ_API_KEY` in your `.env` file to enable Groq AI answers.*"
        )
        return {
            "answer": extracted_answer,
            "citations": citations,
            "provider": "Extractive Grounded Fallback",
        }

    def _call_groq(self, query: str, context: str, scope: str) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }
        system_prompt = (
            "You are LOVA_HR, an expert multi-tenant HR policy chatbot. "
            "Answer user questions concisely and factually, strictly using the provided context chunks for "
            f"{scope}. Do NOT use outside information or guess. If the answer is not in the context, "
            "say 'The policy document in scope does not specify this information.' Always mention source document names."
        )
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"POLICY CONTEXT:\n{context}\n\nUSER QUESTION: {query}"},
            ],
            "temperature": 0.1,
            "max_tokens": 512,
        }
        res = requests.post(url, json=payload, headers=headers, timeout=10)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"].strip()

    def _call_huggingface(self, query: str, context: str, scope: str) -> str:
        url = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json",
        }
        system_prompt = f"You are LOVA_HR assistant. Answer strictly using provided context for {scope}."
        payload = {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}"},
            ],
            "max_tokens": 512,
            "temperature": 0.1,
        }
        res = requests.post(url, json=payload, headers=headers, timeout=12)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"].strip()
