"""
generator.py
Clean LLM Answer Generator for LOVA HR.

Supported local model engines:
  - deepset/roberta-base-squad2   → Extractive QA pipeline (fast, no GPU needed)
  - Qwen/Qwen2.5-0.5B-Instruct   → Tiny CausalLM generative pipeline
  - google/flan-t5-base           → Seq2Seq generative pipeline
  - Qwen/Qwen2.5-1.5B-Instruct   → Larger CausalLM (more capable)
Secondary engine: Cloud HF Router API (requires "Inference Providers" token scope)
Fallback:        High-precision Semantic Policy Processor (rule-based synthesis)

Key features:
  - Strict topic boundary enforcement (refuses out-of-scope questions)
  - Index marker stripping in all outputs via preprocessor
  - Clean structured answer format: Summary + Key Points + Citations
"""
from __future__ import annotations
import os
import re
import concurrent.futures
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

try:
    from .preprocessor import sanitize_chunk, sanitize_query
except ImportError:
    from src.preprocessor import sanitize_chunk, sanitize_query

# ── Model registry ────────────────────────────────────────────────────────────

HF_MODELS: Dict[str, Dict[str, str]] = {
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "label": "Qwen-2.5-0.5B-Instruct (Local)", "icon": "⚡",
        "description": "Local — Tiny 0.5B CausalLM, fastest generative model (Recommended)",
        "engine": "causal_lm",
    },
    "deepset/roberta-base-squad2": {
        "label": "RoBERTa-base-squad2 (Extractive QA)", "icon": "🎯",
        "description": "Local — Fast extractive QA, no GPU required",
        "engine": "extractive_qa",
    },
    "google/flan-t5-base": {
        "label": "Flan-T5-base (Seq2Seq)", "icon": "🌐",
        "description": "Local — Google Flan-T5 text-to-text generation",
        "engine": "seq2seq",
    },
}

DEFAULT_HF_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

_PROVIDERS = {
    "hf_hybrid":  {"label": "HF Hybrid RAG",   "icon": "🔀", "badge_color": "#38bdf8", "badge_bg": "rgba(56,189,248,0.12)", "border": "#38bdf8"},
    "hf_lexical": {"label": "HF Lexical RAG",   "icon": "🔤", "badge_color": "#ec4899", "badge_bg": "rgba(236,72,153,0.12)",  "border": "#ec4899"},
    "hf_semantic":{"label": "HF Semantic RAG",  "icon": "🔵", "badge_color": "#6366f1", "badge_bg": "rgba(99,102,241,0.12)", "border": "#6366f1"},
    "extractive": {"label": "Policy Processor", "icon": "📋", "badge_color": "#6366f1", "badge_bg": "rgba(99,102,241,0.12)", "border": "#6366f1"},
}

# Topic boundary keywords — if query has zero overlap with context keywords, refuse
_OUT_OF_SCOPE_PHRASES = [
    "recipe", "cooking", "sports", "cricket", "football", "weather",
    "stock market", "celebrity", "movie", "song", "game", "geography",
    "capital of", "who is the president", "math problem",
]


_STOP_WORDS = {
    "what", "where", "when", "which", "who", "whom", "whose", "why", "how",
    "does", "do", "did", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "with", "from", "for", "about", "against",
    "between", "into", "through", "during", "before", "after", "above", "below",
    "to", "of", "in", "on", "at", "by", "this", "that", "these", "those",
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "than", "such",
    "policy", "document", "documents", "organization", "company", "employee", "employees",
    "requirement", "requirements", "information", "details", "please", "tell", "give",
}


# ── Multi-Company Comparison Helpers ──────────────────────────────────────────

def _group_chunks_by_company(chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group policy chunks by company name."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for chunk in chunks:
        company = chunk.get("company", "General")
        groups.setdefault(company, []).append(chunk)
    return groups


def _format_company_context(
    company: str, chunks: List[Dict[str, Any]], max_chunks: int = 5
) -> str:
    """Format context for a specific company's chunks."""
    blocks = []
    for chunk in chunks[:max_chunks]:
        raw_text = chunk.get("chunk_text", "")
        clean_text = sanitize_chunk(raw_text)[:1500]
        sec_hdr = chunk.get("section_header", chunk.get("subtopic", ""))
        header = f"Section: {sec_hdr}" if sec_hdr else ""
        blocks.append(f"[{header}]\n{clean_text}")
    return "\n\n".join(blocks)


def _build_company_comparison_prompt(
    query: str,
    company_groups: Dict[str, List[Dict[str, Any]]],
    scope: str = "the selected organizations",
) -> str:
    """Build a prompt for multi-company policy comparison."""
    company_contexts = []
    for company, chunks in list(company_groups.items())[:5]:
        ctx = _format_company_context(company, chunks, max_chunks=3)
        company_contexts.append(f"=== {company} ===\n{ctx}")

    all_context = "\n\n".join(company_contexts)

    prompt = (
        f"You are an HR Policy Assistant analyzing policies across multiple companies: {scope}.\n\n"
        f"The user asked: {query}\n\n"
        f"Based on the following company-specific policy excerpts, provide a structured comparison.\n\n"
        f"COMPANY POLICY EXCERPTS:\n{all_context}\n\n"
        f"Provide your response in this exact format:\n\n"
        f"1. GENERAL POLICY SUMMARY (2-3 sentences synthesizing the common policy across all companies)\n\n"
        f"2. COMPANY-SPECIFIC DIFFERENCES\n"
        f"For each company, explain how their policy differs from the general summary (1-2 sentences each):\n"
        + "\n".join(f"   - {company}: [specific difference or unique aspect]" for company in list(company_groups.keys())[:5]) +
        f"\n\n3. COMPARISON SUMMARY (1-2 sentences comparing the key differences and any recommendations)"
    )
    return prompt


def _generate_multi_company_answer(
    generator: "LLMGenerator",
    query: str,
    chunks: List[Dict[str, Any]],
    hf_model: str = DEFAULT_HF_MODEL,
    use_local_engine: bool = True,
) -> Dict[str, Any]:
    """
    Generate a structured multi-company comparison answer.
    
    Returns dict with:
      - general_answer: synthesized from all companies
      - differences: company-specific variations
      - comparison_summary: bottom-line comparison
      - company_groups: grouped chunks for UI display
    """
    company_groups = _group_chunks_by_company(chunks)
    companies = list(company_groups.keys())

    # If only one company, fall back to standard generation
    if len(companies) <= 1:
        return {"multi_company": False}

    scope = ", ".join(companies[:5]) + ("..." if len(companies) > 5 else "")
    prompt = _build_company_comparison_prompt(query, company_groups, scope)

    # Format all context for the prompt
    all_context_blocks = []
    for company, comp_chunks in company_groups.items():
        for chunk in comp_chunks[:3]:
            raw_text = chunk.get("chunk_text", "")
            clean_text = sanitize_chunk(raw_text)[:2000]
            all_context_blocks.append(f"[{company}] {clean_text}")
    full_context = "\n\n".join(all_context_blocks[:20])

    # Generate answer using the selected engine
    answer_text = ""
    engine_type = HF_MODELS.get(hf_model, {}).get("engine", "causal_lm")

    if use_local_engine and engine_type in ("extractive_qa", "seq2seq", "causal_lm"):
        answer_text = generator._call_local_engine(query, full_context, scope, hf_model)

    if not answer_text:
        answer_text = generator._call_hf_cloud(query, full_context, scope, hf_model)

    # Fallback to rule-based synthesis if LLM fails
    if not answer_text:
        answer_text = _build_multi_company_fallback(query, company_groups)

    # Parse the answer into sections
    sections = _parse_multi_company_response(answer_text, companies)

    return {
        "multi_company": True,
        "company_groups": {k: v[:3] for k, v in company_groups.items()},
        "general_answer": sections.get("general", answer_text[:500]),
        "differences": sections.get("differences", ""),
        "comparison_summary": sections.get("summary", ""),
        "companies": companies,
    }


def _build_multi_company_fallback(
    query: str, company_groups: Dict[str, List[Dict[str, Any]]]
) -> str:
    """Rule-based fallback for multi-company comparison."""
    companies = list(company_groups.keys())[:5]
    lines = [
        "GENERAL POLICY SUMMARY",
        "",
    ]

    # Collect all key phrases from all companies
    all_phrases = []
    for company, chunks in company_groups.items():
        for chunk in chunks[:2]:
            raw = chunk.get("chunk_text", "")
            clean = sanitize_chunk(raw)
            for sentence in re.split(r"[.!?]+", clean):
                sentence = sentence.strip()
                if len(sentence) > 20:
                    all_phrases.append((company, sentence))

    if all_phrases:
        # Synthesize general summary
        seen = set()
        summary_phrases = []
        for company, phrase in all_phrases[:10]:
            if phrase not in seen:
                seen.add(phrase)
                summary_phrases.append(phrase)
        lines.append("Based on the policy documents across all companies:")
        lines.append(" ".join(summary_phrases[:3]) + ".")
    else:
        lines.append("The policy documents contain general information about the requested topic.")

    lines.extend(["", "COMPANY-SPECIFIC DIFFERENCES", ""])

    # Per-company differences
    for company in companies:
        chunks = company_groups[company][:2]
        if chunks:
            raw = chunks[0].get("chunk_text", "")
            clean = sanitize_chunk(raw)[:300]
            first_sentence = re.split(r"[.!?]", clean)[0].strip()
            if first_sentence:
                lines.append(f"- {company}: {first_sentence}.")
            else:
                lines.append(f"- {company}: Follows the standard company policy.")
        else:
            lines.append(f"- {company}: Policy details not available.")

    lines.extend(["", "COMPARISON SUMMARY", ""])
    if len(companies) > 1:
        lines.append(f"Across the {len(companies)} companies analyzed, policies generally follow similar frameworks with company-specific variations in implementation details.")
    else:
        lines.append("Only one company's policy was found in the documents.")

    return "\n".join(lines)


def _parse_multi_company_response(answer_text: str, companies: List[str]) -> Dict[str, str]:
    """Parse LLM response into structured sections."""
    sections = {"general": "", "differences": "", "summary": ""}

    # Try to find section markers
    lines = answer_text.split("\n")
    current_section = "general"
    section_content = {"general": [], "differences": [], "summary": []}

    for line in lines:
        line_lower = line.lower().strip()
        if "general" in line_lower and ("summary" in line_lower or "policy" in line_lower):
            current_section = "general"
            continue
        elif "difference" in line_lower or "variation" in line_lower or "company-specific" in line_lower:
            current_section = "differences"
            continue
        elif "comparison" in line_lower and "summary" in line_lower:
            current_section = "summary"
            continue

        if line.strip():
            section_content[current_section].append(line)

    # Join sections
    for key in sections:
        sections[key] = "\n".join(section_content[key]).strip()

    # Fallback: if parsing failed, use whole text as general
    if not sections["general"] and not sections["differences"]:
        sections["general"] = answer_text[:800]

    return sections


def _is_out_of_scope(query: str, context: str) -> bool:
    """
    Returns True if the query is clearly outside the provided document context.
    Uses a two-pass check:
      1. Explicit out-of-scope keyword match
      2. Key subject query term overlap (excluding stop words)
    """
    q_lower = query.lower()
    for phrase in _OUT_OF_SCOPE_PHRASES:
        if phrase in q_lower:
            return True

    # Extract non-stop words of length >= 3
    query_words = set(re.findall(r"\b\w{3,}\b", q_lower)) - _STOP_WORDS
    context_words = set(re.findall(r"\b\w{3,}\b", context.lower())) - _STOP_WORDS

    if query_words and context_words:
        overlap = query_words & context_words
        if not overlap:
            print(f"[LLMGenerator] Out-of-scope detected: Query terms {query_words} not found in context.")
            return True
    return False


def _build_system_prompt(scope: str) -> str:
    return (
        f"You are a precise HR Policy Assistant for {scope}. "
        "Answer ONLY based on the provided policy context. "
        "If the answer is not explicitly or implicitly present in the context, "
        "say: 'This information is not available in the provided policy documents.' "
        "Do NOT guess, hallucinate, or use external knowledge. "
        "Do NOT include index numbers, list markers, or citation brackets in your answer. "
        "Format your answer as clear, natural prose."
    )


def _process_context_to_natural_prose(
    chunks: List[Dict[str, Any]], query: str, scope: str = "the organization"
) -> str:
    """
    Rule-based synthesis: converts policy chunks into clean, structured prose.
    Applied when LLM APIs are unavailable.
    All index markers stripped via preprocessor.
    """
    if not chunks:
        return "I cannot find specific details regarding that in the provided policy documents."

    items: List[str] = []
    unique_cites: List[str] = []

    for chunk in chunks[:3]:
        f_name  = chunk.get("filename", "Policy Document")
        page_num = chunk.get("page_number", chunk.get("page", 1))
        sec_hdr  = chunk.get("section_header", "")
        cite = f"{f_name} - {sec_hdr} (p. {page_num})" if sec_hdr else f"{f_name} (p. {page_num})"
        if cite not in unique_cites:
            unique_cites.append(cite)

        raw = chunk.get("chunk_text", "").strip()
        clean = sanitize_chunk(raw)
        for line in clean.split("\n"):
            line = line.strip()
            if len(line) >= 10:
                items.append(line)

    # Filter lines that match key query terms if query words exist
    q_terms = set(re.findall(r"\b\w{3,}\b", query.lower())) - _STOP_WORDS
    matched_items = []
    if q_terms:
        for item in items:
            item_terms = set(re.findall(r"\b\w{3,}\b", item.lower()))
            if q_terms & item_terms:
                matched_items.append(item)
    
    selected_items = matched_items if matched_items else items

    if not selected_items:
        return f"The provided policy documents do not contain specific information regarding '{query}'."

    summary = " ".join(selected_items[:3])
    key_points = []
    seen = set()
    for item in selected_items[:8]:
        clean = item.lstrip("*-• ").strip()
        if clean and clean not in seen:
            seen.add(clean)
            key_points.append(f"- {clean}")

    sections = [
        "### Summary\n" + summary,
        "### Key Points\n" + ("\n".join(key_points) if key_points else "- Refer to official policy documents."),
    ]
    if unique_cites:
        sections.append("### Reference Sources\n" + "\n".join(f"- {c}" for c in unique_cites))

    return "\n\n".join(sections)


def _format_context(chunks: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
    """Format top chunks into clean reference blocks for LLM prompts."""
    blocks: List[str] = []
    citations: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks[:5], 1):
        f_name   = chunk.get("filename", "Policy Document")
        page_num = chunk.get("page_number", chunk.get("page", 1))
        sec_hdr  = chunk.get("section_header", chunk.get("subtopic", ""))
        raw_text = chunk.get("chunk_text", "")
        clean_text = sanitize_chunk(raw_text)[:2500]

        hdr = f"--- [Document: {f_name}, Page: {page_num}]"
        if sec_hdr:
            hdr = f"--- [Document: {f_name}, Section: {sec_hdr}, Page: {page_num}]"
        blocks.append(f"{hdr} ---\n{clean_text}")

        citations.append({
            "rank": idx,
            "company": chunk.get("company", "General"),
            "subfolder": chunk.get("subfolder", "General"),
            "filename": f_name,
            "page": page_num,
            "section_header": sec_hdr,
            "score": chunk.get("reranker_score", chunk.get("rrf_score", chunk.get("bm25_score", 0.0))),
            "source_file": chunk.get("source_file", ""),
        })

    return "\n\n".join(blocks), citations


class LLMGenerator:
    """
    Clean LLM Answer Generator with strict topic boundaries.

    Supports three local engine types:
      - extractive_qa : deepset/roberta-base-squad2 (extractive span extraction)
      - causal_lm     : Qwen2.5-0.5B / 1.5B (CausalLM chat template generation)
      - seq2seq       : google/flan-t5-base (text-to-text generation)
    Secondary: Cloud HF Router API
    Fallback:  Rule-based Semantic Policy Processor
    """

    def __init__(self):
        self._local_model = None
        self._local_tokenizer = None
        self._local_model_id: Optional[str] = None
        self._qa_pipeline = None
        self._qa_model_id: Optional[str] = None
        self._seq2seq_pipeline = None
        self._seq2seq_model_id: Optional[str] = None

    def _get_hf_token(self) -> str:
        try:
            from dotenv import load_dotenv
            root = Path(__file__).resolve().parent.parent
            env = root / ".env"
            load_dotenv(dotenv_path=env if env.exists() else None, override=True)
        except Exception:
            pass
        return os.getenv("HF_TOKEN", "").strip()

    # ── Local transformers engine ─────────────────────────────────────────────

    def _load_local_model(self, model_id: str) -> bool:
        """Load local Qwen2.5-1.5B model via transformers (lazy, cached)."""
        if self._local_model_id == model_id and self._local_model is not None:
            return True
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            print(f"[LLMGenerator] Loading local model: {model_id}")
            self._local_tokenizer = AutoTokenizer.from_pretrained(
                model_id, token=self._get_hf_token() or None
            )
            self._local_model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto",
                torch_dtype=torch.float32,
                token=self._get_hf_token() or None,
            )
            self._local_model_id = model_id
            print(f"[LLMGenerator] Local model loaded: {model_id}")
            return True
        except Exception as exc:
            print(f"[LLMGenerator] Local model load failed ({model_id}): {exc}")
            return False

    def _call_local_causal_lm(
        self, query: str, context: str = "", scope: str = "the organization",
        model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens: int = 256,
    ) -> str:
        """Generate answer using local AutoModelForCausalLM."""
        if not self._load_local_model(model_id):
            return ""
        try:
            import torch
            system_prompt = _build_system_prompt(scope)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}" if context else query},
            ]
            inputs = self._local_tokenizer.apply_chat_template(
                messages, add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt",
            ).to(self._local_model.device)
            with torch.no_grad():
                outputs = self._local_model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    do_sample=False, temperature=1.0,
                )
            answer = self._local_tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
            ).strip()
            return sanitize_chunk(answer)  # Strip any index markers from LLM output
        except Exception as exc:
            print(f"[LLMGenerator] Local generation error: {exc}")
            return ""

    def generate_local_causal_lm(
        self, query: str, context: str = "", scope: str = "the organization",
        model_id: str = "Qwen/Qwen2.5-1.5B-Instruct", max_new_tokens: int = 256,
    ) -> str:
        """Public alias for _call_local_causal_lm."""
        return self._call_local_causal_lm(query, context, scope, model_id, max_new_tokens)

    # ── Extractive QA engine (deepset/roberta-base-squad2) ────────────────────

    def _call_extractive_qa(
        self, query: str, context: str,
        model_id: str = "deepset/roberta-base-squad2",
    ) -> str:
        """
        Use a HuggingFace extractive QA pipeline to find the exact answer span
        in the provided context. No GPU required — runs on CPU.
        Model: deepset/roberta-base-squad2
        """
        if self._qa_model_id != model_id or self._qa_pipeline is None:
            try:
                from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline as hf_pipeline
                print(f"[LLMGenerator] Loading QA pipeline: {model_id}")
                token_val = self._get_hf_token() or None
                tokenizer = AutoTokenizer.from_pretrained(model_id, token=token_val)
                model = AutoModelForQuestionAnswering.from_pretrained(model_id, token=token_val)
                self._qa_pipeline = hf_pipeline(
                    "question-answering",
                    model=model,
                    tokenizer=tokenizer,
                    device=-1,  # Force CPU
                )
                self._qa_model_id = model_id
                print(f"[LLMGenerator] QA pipeline loaded: {model_id}")
            except Exception as exc:
                print(f"[LLMGenerator] QA pipeline load failed ({model_id}): {exc}")
                return ""

        if not context or not query:
            return ""

        try:
            # Split context into chunks of max 512 tokens and QA over each
            context_chunks = [context[i:i+1500] for i in range(0, min(len(context), 4500), 1500)]
            best_answer = ""
            best_score  = -1.0
            for ctx_chunk in context_chunks:
                if not ctx_chunk.strip():
                    continue
                result = self._qa_pipeline(
                    question=query,
                    context=ctx_chunk,
                    max_answer_len=200,
                    handle_impossible_answer=True,
                )
                if result and result.get("score", 0) > best_score:
                    best_score  = result["score"]
                    best_answer = result.get("answer", "")

            if not best_answer or best_score < 0.01:
                return ""
            return sanitize_chunk(best_answer)
        except Exception as exc:
            print(f"[LLMGenerator] Extractive QA error: {exc}")
            return ""

    # ── Seq2Seq engine (google/flan-t5-base) ──────────────────────────────────

    def _call_seq2seq(
        self, query: str, context: str, scope: str,
        model_id: str = "google/flan-t5-base",
        max_new_tokens: int = 200,
    ) -> str:
        """
        Use a HuggingFace text2text-generation pipeline with google/flan-t5-base.
        Constructs a structured prompt and generates a grounded answer.
        """
        if self._seq2seq_model_id != model_id or self._seq2seq_pipeline is None:
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline as hf_pipeline
                print(f"[LLMGenerator] Loading Seq2Seq pipeline: {model_id}")
                token_val = self._get_hf_token() or None
                tokenizer = AutoTokenizer.from_pretrained(model_id, token=token_val)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id, token=token_val)
                self._seq2seq_pipeline = hf_pipeline(
                    "text2text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=-1,  # CPU
                )
                self._seq2seq_model_id = model_id
                print(f"[LLMGenerator] Seq2Seq pipeline loaded: {model_id}")
            except Exception as exc:
                print(f"[LLMGenerator] Seq2Seq pipeline load failed ({model_id}): {exc}")
                return ""

        try:
            # Flan-T5 works best with an explicit instruction prefix
            prompt = (
                f"Answer the following HR policy question based only on the provided context.\n"
                f"If the answer is not in the context, say 'Not found in policy documents'.\n\n"
                f"Context: {context[:1024]}\n\n"
                f"Question: {query}\n\n"
                f"Answer:"
            )
            result = self._seq2seq_pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            if result and len(result) > 0:
                answer = result[0].get("generated_text", "").strip()
                return sanitize_chunk(answer)
            return ""
        except Exception as exc:
            print(f"[LLMGenerator] Seq2Seq generation error: {exc}")
            return ""

    # ── Unified local engine dispatcher ──────────────────────────────────────

    def _call_local_engine(
        self, query: str, context: str, scope: str, model_id: str,
    ) -> str:
        """
        Route to the correct local inference engine based on model_id engine type.
        Order of engine selection:
          1. extractive_qa → deepset/roberta-base-squad2
          2. seq2seq       → google/flan-t5-base
          3. causal_lm     → Qwen2.5-0.5B / 1.5B (chat template)
          4. cloud         → skip local, use cloud API
        """
        engine = HF_MODELS.get(model_id, {}).get("engine", "causal_lm")

        if engine == "extractive_qa":
            return self._call_extractive_qa(query, context, model_id)
        elif engine == "seq2seq":
            return self._call_seq2seq(query, context, scope, model_id)
        elif engine == "causal_lm":
            return self._call_local_causal_lm(query, context, scope, model_id)
        else:
            # cloud / vision — handled by cloud API
            return ""

    # ── Cloud HF Router engine ────────────────────────────────────────────────

    def _call_hf_cloud(
        self, query: str, context: str, scope: str,
        model_id: str = DEFAULT_HF_MODEL,
        image_url: Optional[str] = None,
    ) -> str:
        """Call the HF Router cloud API."""
        token = self._get_hf_token()
        if not token:
            return ""

        system_prompt = _build_system_prompt(scope)
        user_content: Any
        if image_url:
            user_content = [
                {"type": "image", "url": image_url},
                {"type": "text",  "text": f"Context:\n{context}\n\nQuestion: {query}" if context else query},
            ]
        else:
            user_content = f"Context:\n{context}\n\nQuestion: {query}" if context else query

        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ],
            "max_tokens": 512,
            "temperature": 0.0,
        }
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        endpoints = [
            "https://router.huggingface.co/hf-inference/v1/chat/completions",
            f"https://router.huggingface.co/models/{model_id}/v1/chat/completions",
        ]
        for url in endpoints:
            try:
                res = requests.post(url, json=payload, headers=headers, timeout=10)
                if res.status_code == 200:
                    data = res.json()
                    if data.get("choices"):
                        ans = data["choices"][0]["message"]["content"].strip()
                        return sanitize_chunk(ans)
                elif res.status_code == 403:
                    print("[LLMGenerator] HF 403: Token lacks Inference Providers scope.")
                    return ""
                res.raise_for_status()
            except requests.exceptions.RequestException as exc:
                print(f"[LLMGenerator] HF cloud error: {exc}")
                continue
        return ""

    # ── Multimodal engine ─────────────────────────────────────────────────────

    def generate_multimodal_answer(
        self,
        query: str,
        image_url: str,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        context: str = "",
        use_local_transformers: bool = False,
    ) -> str:
        """Generate answer from image + text query using vision-language model."""
        if use_local_transformers:
            try:
                from transformers import AutoProcessor, AutoModelForVision2Seq
                import torch, requests as rq
                from PIL import Image
                from io import BytesIO
                print(f"[LLMGenerator] Loading VL model: {model_id}")
                processor = AutoProcessor.from_pretrained(model_id, token=self._get_hf_token() or None)
                model = AutoModelForVision2Seq.from_pretrained(
                    model_id, device_map="auto", torch_dtype=torch.float16,
                    token=self._get_hf_token() or None,
                )
                img_resp = rq.get(image_url, timeout=10)
                image = Image.open(BytesIO(img_resp.content)).convert("RGB")
                messages = [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Context: {context}\nQuestion: {query}" if context else query},
                ]}]
                inputs = processor.apply_chat_template(
                    messages, images=[image], add_generation_prompt=True, return_tensors="pt"
                ).to(model.device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=256)
                answer = processor.decode(outputs[0], skip_special_tokens=True).strip()
                return sanitize_chunk(answer)
            except Exception as exc:
                print(f"[LLMGenerator] VL local error: {exc}. Falling back to cloud.")
        return self._call_hf_cloud(query, context, "the organization", model_id=model_id, image_url=image_url)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        company: Optional[str] = None,
        max_providers: int = 1,
        hf_model: str = DEFAULT_HF_MODEL,
        rag_mode: str = "hybrid",
        semantic_chunks: Optional[List[Dict[str, Any]]] = None,
        lexical_chunks: Optional[List[Dict[str, Any]]] = None,
        use_local_engine: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a high-precision, semantically grounded answer.

        Steps:
          1. Sanitize query (strip index markers via preprocessor)
          2. Topic boundary check — refuse out-of-scope questions
          3. Format context from top retrieved chunks
          4. Dispatch to local engine based on model type:
               extractive_qa → deepset/roberta-base-squad2
               seq2seq       → google/flan-t5-base
               causal_lm     → Qwen2.5-0.5B / 1.5B
          5. Cloud HF Router API fallback
          6. Rule-based Semantic Policy Processor (final fallback)
          
        For "All Companies" mode (company=None), generates a structured
        multi-company comparison with general answer, differences, and summary.
        """
        clean_query = sanitize_query(query)
        active_chunks = chunks or semantic_chunks or lexical_chunks or []
        scope = company if (company and company != "All Companies") else "the selected organization"

        # ── Multi-Company Comparison Mode ──────────────────────────────────
        if not company or company == "All Companies":
            multi_result = _generate_multi_company_answer(
                generator=self,
                query=clean_query,
                chunks=active_chunks,
                hf_model=hf_model,
                use_local_engine=use_local_engine,
            )
            if multi_result.get("multi_company"):
                # Multi-company mode: return structured comparison
                company_groups = multi_result.get("company_groups", {})
                all_company_chunks = []
                for comp_chunks in company_groups.values():
                    all_company_chunks.extend(comp_chunks)
                _, citations = _format_context(all_company_chunks[:15])

                general_answer = multi_result.get("general_answer", "")
                differences = multi_result.get("differences", "")
                comparison_summary = multi_result.get("comparison_summary", "")

                answers = [
                    {
                        "provider_key": "multi_company_general",
                        "answer": general_answer,
                        "label": "General Policy Summary",
                        "icon": "📋",
                        "badge_color": "#10b981",
                        "section_type": "general",
                    },
                ]
                if differences:
                    answers.append({
                        "provider_key": "multi_company_diff",
                        "answer": differences,
                        "label": "Company Differences",
                        "icon": "🔀",
                        "badge_color": "#6366f1",
                        "section_type": "differences",
                    })
                if comparison_summary:
                    answers.append({
                        "provider_key": "multi_company_summary",
                        "answer": comparison_summary,
                        "label": "Comparison Summary",
                        "icon": "📊",
                        "badge_color": "#f59e0b",
                        "section_type": "summary",
                    })

                return {
                    "answers": answers,
                    "citations": citations,
                    "answer": general_answer,
                    "provider": "Multi-Company Comparison",
                    "hf_model_used": hf_model,
                    "rag_mode": rag_mode,
                    "multi_company": True,
                    "company_groups": company_groups,
                    "companies": multi_result.get("companies", []),
                }
            # Fallback to single-company format if multi-company failed
            scope = "the selected organization"

        if not active_chunks:
            msg = "No relevant policy documents were found in scope to answer this question."
            return {"answers": [{"provider_key": "extractive", "answer": msg}],
                    "answer": msg, "citations": [], "provider": "none"}

        # Format context
        if rag_mode == "fullylexical":
            ctx_chunks = lexical_chunks or active_chunks
        elif rag_mode == "semantic":
            ctx_chunks = semantic_chunks or active_chunks
        else:
            ctx_chunks = active_chunks
        formatted_context, citations = _format_context(ctx_chunks)

        # Topic boundary check
        if _is_out_of_scope(clean_query, formatted_context):
            msg = (
                "This question appears to be outside the scope of the provided HR policy documents. "
                "I can only answer questions directly related to the policies in the uploaded documents."
            )
            return {"answers": [{"provider_key": "extractive", "answer": msg, "label": "Topic Boundary",
                                  "icon": "🚫", "badge_color": "#ef4444"}],
                    "answer": msg, "citations": [], "provider": "Topic Boundary",
                    "rag_mode": rag_mode, "hf_model_used": hf_model}

        prose_fallback = _process_context_to_natural_prose(active_chunks, clean_query, scope)
        answers: List[Dict[str, Any]] = []

        def _try_generate(key: str, context: str) -> Optional[Dict[str, Any]]:
            ans_text = ""
            # 1. Try local engine (extractive QA / seq2seq / causal LM based on model type)
            engine_type = HF_MODELS.get(hf_model, {}).get("engine", "causal_lm")
            if use_local_engine and engine_type in ("extractive_qa", "seq2seq", "causal_lm"):
                ans_text = self._call_local_engine(clean_query, context, scope, hf_model)
            # 2. Cloud API fallback
            if not ans_text:
                ans_text = self._call_hf_cloud(clean_query, context, scope, hf_model)

            if ans_text:
                model_meta  = HF_MODELS.get(hf_model, {})
                model_label = model_meta.get("label", hf_model.split("/")[-1])
                icon        = model_meta.get("icon", "🤖")
                color_map   = {
                    "extractive_qa": "#10b981",  # emerald
                    "causal_lm":     "#38bdf8",  # sky blue
                    "seq2seq":       "#6366f1",  # indigo
                    "cloud":         "#ec4899",  # pink
                    "vision":        "#f59e0b",  # amber
                }
                badge_color = color_map.get(engine_type, "#38bdf8")

                return {
                    "provider_key": key,
                    "answer": ans_text,
                    "label": model_label,
                    "icon": icon,
                    "badge_color": badge_color,
                    "engine": engine_type,
                }
            return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures: Dict[str, Any] = {}
            if rag_mode == "semantic_hybrid":
                sem_ctx, _ = _format_context(semantic_chunks or active_chunks)
                hyb_ctx, _ = _format_context(chunks or active_chunks)
                futures["hf_semantic"] = pool.submit(_try_generate, "hf_semantic", sem_ctx)
                futures["hf_hybrid"]   = pool.submit(_try_generate, "hf_hybrid",   hyb_ctx)
            elif rag_mode == "fullylexical":
                lex_ctx, _ = _format_context(lexical_chunks or active_chunks)
                futures["hf_lexical"] = pool.submit(_try_generate, "hf_lexical", lex_ctx)
            elif rag_mode == "semantic":
                sem_ctx, _ = _format_context(semantic_chunks or active_chunks)
                futures["hf_semantic"] = pool.submit(_try_generate, "hf_semantic", sem_ctx)
            else:
                futures["hf_hybrid"] = pool.submit(_try_generate, "hf_hybrid", formatted_context)

            for key in ["hf_hybrid", "hf_semantic", "hf_lexical"]:
                if key in futures:
                    result = futures[key].result()
                    if result:
                        answers.append(result)

        if not answers:
            answers.append({"provider_key": "extractive", "answer": prose_fallback,
                             "label": _PROVIDERS["extractive"]["label"],
                             "icon": _PROVIDERS["extractive"]["icon"],
                             "badge_color": _PROVIDERS["extractive"]["badge_color"]})

        answers = answers[:max(1, max_providers)]
        best = answers[0]
        return {
            "answers": answers,
            "citations": citations,
            "answer": best["answer"],
            "provider": best.get("label", best["provider_key"]),
            "hf_model_used": hf_model,
            "rag_mode": rag_mode,
        }

    # Legacy alias
    def _call_huggingface(self, query: str, context: str, scope: str) -> str:
        return self._call_hf_cloud(query, context, scope)
