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
    from .preprocessor import sanitize_chunk, sanitize_query, clean_conversational_response
    from .config import CLOUD_ONLY_MODE
except ImportError:
    from src.preprocessor import sanitize_chunk, sanitize_query, clean_conversational_response
    from src.config import CLOUD_ONLY_MODE

# ── Model registry ────────────────────────────────────────────────────────────

HF_MODELS: Dict[str, Dict[str, str]] = {
    "Qwen/Qwen2.5-3B-Instruct": {
        "label": "Qwen-2.5-3B-Instruct (Local)", "icon": "⚡",
        "description": "Local — 3B CausalLM, high-accuracy conversational generative model (Default)",
        "engine": "causal_lm",
    },
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "label": "Qwen-2.5-0.5B-Instruct (Local)", "icon": "⚡",
        "description": "Local — Tiny 0.5B CausalLM, ultra-fast generative model",
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

DEFAULT_HF_MODEL = "deepset/roberta-base-squad2"

_PROVIDERS = {
    "hf_hybrid":  {"label": "HF Hybrid RAG",   "icon": "🔀", "badge_color": "#38bdf8", "badge_bg": "rgba(56,189,248,0.12)", "border": "#38bdf8"},
    "hf_lexical": {"label": "HF Lexical RAG",   "icon": "🔤", "badge_color": "#ec4899", "badge_bg": "rgba(236,72,153,0.12)",  "border": "#ec4899"},
    "hf_semantic":{"label": "HF Semantic RAG",  "icon": "🔵", "badge_color": "#6366f1", "badge_bg": "rgba(99,102,241,0.12)", "border": "#6366f1"},
    "extractive": {"label": "Policy Processor", "icon": "📋", "badge_color": "#6366f1", "badge_bg": "rgba(99,102,241,0.12)", "border": "#6366f1"},
}

# Topic boundary keywords — if query has zero overlap with context keywords, refuse
_STOP_WORDS = {
    "what", "where", "when", "which", "who", "whom", "whose", "why", "how",
    "does", "do", "did", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "with", "from", "for", "about", "against",
    "between", "into", "through", "during", "before", "after", "above", "below",
    "to", "of", "in", "on", "at", "by", "this", "that", "these", "those",
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "than", "such",
    "policy", "document", "documents", "organization", "company", "employee", "employees",
    "requirement", "requirements", "information", "details", "please", "tell", "give",
    "hello", "hi", "hey", "greetings", "good", "morning", "afternoon", "evening",
    "thanks", "thank", "you", "can", "could", "would", "should", "will", "i", "my",
    "me", "we", "our", "us", "am", "want", "like", "know", "find", "help", "ask",
    "asking", "wondering", "regards", "regarding", "concerning",
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
        company_contexts.append(f"Company {company}:\n{ctx}")

    all_context = "\n\n".join(company_contexts)

    prompt = (
        f"You are a warm, helpful HR colleague comparing policies across companies: {scope}.\n\n"
        f"An employee asked: {query}\n\n"
        f"POLICY EXCERPTS:\n{all_context}\n\n"
        f"How to respond:\n"
        f"1. Write like you're talking to a friend — warm, natural, conversational.\n"
        f"2. Weave the comparison into your explanation. Like: 'At TCS you get X, but Infosys offers Y.'\n"
        f"3. Bold key terms naturally: **26 weeks**, **5 lakh coverage**, **60 days notice**.\n"
        f"4. Start new points on new lines.\n"
        f"5. No robotic phrases. No 'The document states...'.\n"
        f"6. Only use tables or lists if the user specifically asks for them."
    )
    return prompt


def _generate_multi_company_answer(
    generator: "LLMGenerator",
    query: str,
    chunks: List[Dict[str, Any]],
    hf_model: str = DEFAULT_HF_MODEL,
    use_local_engine: bool = False,
) -> Dict[str, Any]:
    """Generate a structured multi-company comparison answer."""
    company_groups = _group_chunks_by_company(chunks)
    companies = list(company_groups.keys())

    if len(companies) <= 1:
        return {"multi_company": False}

    scope = ", ".join(companies[:5]) + ("..." if len(companies) > 5 else "")
    prompt = _build_company_comparison_prompt(query, company_groups, scope)

    all_context_blocks = []
    for company, comp_chunks in company_groups.items():
        for chunk in comp_chunks[:3]:
            raw_text = chunk.get("chunk_text", "")
            clean_text = sanitize_chunk(raw_text)[:2000]
            all_context_blocks.append(f"[{company}] {clean_text}")
    full_context = "\n\n".join(all_context_blocks[:20])

    answer_text = ""
    engine_type = HF_MODELS.get(hf_model, {}).get("engine", "causal_lm")

    if CLOUD_ONLY_MODE and use_local_engine:
        print("[LLMGenerator] CLOUD_ONLY_MODE enabled — forcing cloud API for multi-company")
        use_local_engine = False

    if engine_type == "extractive_qa":
        answer_text = generator._call_extractive_qa(query, full_context, hf_model)
        if not answer_text:
            answer_text = _build_multi_company_fallback(query, company_groups)
    elif use_local_engine and engine_type in ("seq2seq", "causal_lm"):
        answer_text = generator._call_local_engine(query, full_context, scope, hf_model)

    if not answer_text and engine_type != "extractive_qa":
        answer_text = generator._call_hf_cloud(query, full_context, scope, hf_model)

    if not answer_text:
        answer_text = _build_multi_company_fallback(query, company_groups)

    sections = _parse_multi_company_response(answer_text, companies)

    return {
        "multi_company": True,
        "company_groups": {k: v[:3] for k, v in company_groups.items()},
        "general_answer": clean_conversational_response(sections.get("general", answer_text[:500])),
        "differences": clean_conversational_response(sections.get("differences", "")),
        "comparison_summary": clean_conversational_response(sections.get("summary", "")),
        "companies": companies,
    }


def _build_multi_company_fallback(
    query: str, company_groups: Dict[str, List[Dict[str, Any]]]
) -> str:
    """Rule-based conversational fallback for multi-company comparison."""
    companies = list(company_groups.keys())[:5]
    sentences = ["Based on the policy documents across the companies:"]

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
        seen = set()
        for company, phrase in all_phrases[:6]:
            clean_p = re.sub(r"^[\s*•\-#\d\.\)]+", "", phrase).strip()
            if clean_p and clean_p not in seen:
                seen.add(clean_p)
                sentences.append(f"For {company}, {clean_p}.")
    else:
        sentences.append("The policy documents contain general information regarding the requested topic.")

    if len(companies) > 1:
        sentences.append(f"Across the {len(companies)} companies analyzed, policies generally follow similar frameworks with company-specific variations.")

    res = " ".join(sentences)
    return clean_conversational_response(res)

_OUT_OF_SCOPE_PHRASES = [
    "recipe", "cooking", "sports", "cricket", "football", "weather",
    "stock market", "celebrity", "movie", "song", "game", "geography",
    "capital of", "who is the president", "math problem",
]



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
        f"You are a friendly, helpful HR colleague explaining company policy for {scope} directly to an employee.\n\n"
        "### HOW TO RESPOND\n"
        "Write like you're talking to a friend at work. Be warm, clear, and natural.\n\n"
        "Rules:\n"
        "1. Conversational Tone: Write like a real person talking, not a document. Use simple, friendly language.\n"
        "2. Natural Flow: One idea flows into the next. No bullet points, no tables, no headers unless specifically asked.\n"
        "3. Bold key terms naturally: **5 lakh coverage**, **26 weeks maternity**, **60 days notice period**.\n"
        "4. Start new points on new lines for readability.\n"
        "5. If comparing two things, weave the comparison into your explanation naturally. Like: 'TCS gives you 26 weeks of maternity leave, while Infosys offers 24 weeks.'\n"
        "6. No robotic phrases: Never say 'The document states', 'According to policy', or 'As per the guidelines'.\n"
        "7. Be helpful: Add context that helps the employee understand, like what applies to them.\n\n"
        "Only use tables, lists, or structured formats if the user specifically asks for them."
    )


def _process_context_to_natural_prose(
    chunks: List[Dict[str, Any]], query: str, scope: str = "the organization"
) -> str:
    """
    Rule-based synthesis: converts policy chunks into clean, human-like conversational prose.
    Strips out markdown markers, bullets, headers, and typographical noise.
    """
    if not chunks:
        return "I checked our policy documents, but I couldn't find specific details regarding your request."

    items: List[str] = []
    seen_items: set = set()
    for chunk in chunks[:5]:
        raw = chunk.get("chunk_text", "").strip()
        clean = sanitize_chunk(raw)
        for line in clean.split("\n"):
            line = line.strip()
            if len(line) < 15:
                continue
            # Skip section numbers and headers like "3 3 Workplace Diversity" or "Section 4.2"
            if re.match(r"^\d+[\s\.]?\d*\s+[A-Z]", line) and len(line) < 60:
                continue
            # Skip lines that are just company names or document titles
            if re.match(r"^(?:For\s+)?[\w\s]+(?:Limited|Corp|Inc|Pvt|Ltd)\.?\s*$", line, re.IGNORECASE):
                continue
            # Remove "For [Company] ," prefix patterns
            line = re.sub(r"(?i)^For\s+[\w\s]+?,\s*", "", line).strip()
            # Remove "Human Rights Policy Statement" and similar doc titles mid-sentence
            line = re.sub(r"(?i)\b(?:Human Rights Policy Statement|Policy Statement|Workplace Diversity)\s*", "", line).strip()
            # Deduplicate
            line_lower = line.lower().strip()
            if line_lower not in seen_items:
                seen_items.add(line_lower)
                items.append(line)

    q_terms = set(re.findall(r"\b\w{3,}\b", query.lower())) - _STOP_WORDS
    matched_items = []
    if q_terms:
        for item in items:
            item_terms = set(re.findall(r"\b\w{3,}\b", item.lower()))
            if q_terms & item_terms:
                matched_items.append(item)

    selected_items = matched_items if matched_items else items

    if not selected_items:
        return f"I searched through the policy documents for {scope}, but I couldn't find details regarding your inquiry."

    clean_sentences = []
    seen = set()
    seen_lower_map = {}
    for item in selected_items[:6]:
        clean = re.sub(r"^[\s*•\-#\d\.\)]+", "", item).strip()
        # Translate corporate jargon into friendly human language
        clean = re.sub(r"(?i)associates are eligible for a balanced pool of leaves to maintain professional and personal equilibrium:?", "", clean).strip()
        clean = re.sub(r"(?i)immediate personal interventions", "urgent personal matters", clean)
        clean = re.sub(r"(?i)repatriation tracking", "a plan for your return", clean)
        clean = re.sub(r"(?i)accrued month-on-month", "earned each month", clean)
        clean = re.sub(r"(?i)requiring structured manager sign-off", "with manager approval", clean)
        clean = re.sub(r"(?i)allocated for unexpected medical occurrences", "set aside for medical needs", clean)
        clean = re.sub(r"(?i)provided to high-performing long-term associates", "offered to eligible team members", clean)
        # Remove orphaned company prefixes
        clean = re.sub(r"(?i)^(?:For\s+)?(?:TCS|Infosys|Wipro|Tata|Accenture)\s*,?\s*", "", clean).strip()
        # Skip empty or too-short lines
        if len(clean) < 10:
            continue
        # Ensure sentence ends with punctuation
        if not clean.endswith((".", "!", "?")):
            clean += "."
        # Deduplicate by checking if this sentence is a subset of an already-seen one
        clean_lower = clean.lower().strip()
        is_dup = False
        for prev_lower in seen_lower_map.values():
            if clean_lower in prev_lower or prev_lower in clean_lower:
                is_dup = True
                break
        if not is_dup:
            seen.add(clean)
            seen_lower_map[clean] = clean_lower
            clean_sentences.append(clean)

    if not clean_sentences:
        return f"I found relevant information in the policy documents for {scope}, but I couldn't form a complete answer. Please try rephrasing your question."

    prose = " ".join(clean_sentences)
    return clean_conversational_response(prose)


def _format_context(chunks: List[Dict[str, Any]], max_chunks: int = 3, max_chars_per_chunk: int = 800) -> tuple[str, List[Dict[str, Any]]]:
    """Format top chunks into clean, concise reference blocks for ultra-fast LLM prompts."""
    blocks: List[str] = []
    citations: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks[:max_chunks], 1):
        f_name   = chunk.get("filename", "Policy Document")
        page_num = chunk.get("page_number", chunk.get("page", 1))
        sec_hdr  = chunk.get("section_header", chunk.get("subtopic", ""))
        raw_text = chunk.get("chunk_text", "")
        clean_text = sanitize_chunk(raw_text)[:max_chars_per_chunk]

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
            token_val = self._get_hf_token() or None
            self._local_tokenizer = AutoTokenizer.from_pretrained(
                model_id, token=token_val
            )
            # Use device_map="auto" only when CUDA is available;
            # on CPU-only systems it causes accelerate to attempt disk offload and crash.
            if torch.cuda.is_available():
                self._local_model = AutoModelForCausalLM.from_pretrained(
                    model_id, device_map="auto",
                    dtype=torch.float16,
                    token=token_val,
                )
            else:
                self._local_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    token=token_val,
                ).to("cpu")
            self._local_model_id = model_id
            print(f"[LLMGenerator] Local model loaded: {model_id}")
            return True
        except Exception as exc:
            print(f"[LLMGenerator] Local model load failed ({model_id}): {exc}")
            return False

    def _call_local_causal_lm(
        self, query: str, context: str = "", scope: str = "the organization",
        model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens: int = 512,
    ) -> str:
        """Generate answer using local AutoModelForCausalLM."""
        if not self._load_local_model(model_id):
            return ""
        try:
            import torch
            try:
                torch.set_num_threads(min(8, os.cpu_count() or 4))
            except Exception:
                pass
            system_prompt = _build_system_prompt(scope)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Context:\n{context[:1500]}\n\nQuestion: {query}" if context else query},
            ]
            encoded = self._local_tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            if isinstance(encoded, torch.Tensor):
                input_ids = encoded.to(self._local_model.device)
                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                }
            else:
                inputs = {
                    k: (v.to(self._local_model.device) if hasattr(v, "to") else torch.tensor(v).to(self._local_model.device))
                    for k, v in encoded.items()
                }

            if "attention_mask" not in inputs:
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

            pad_id = getattr(self._local_tokenizer, "pad_token_id", None) or getattr(self._local_tokenizer, "eos_token_id", None)
            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": min(max_new_tokens, 512),
                "do_sample": False,
                "use_cache": True,
            }
            if pad_id is not None:
                gen_kwargs["pad_token_id"] = pad_id

            with torch.no_grad():
                outputs = self._local_model.generate(**inputs, **gen_kwargs)

            input_length = inputs["input_ids"].shape[-1]
            answer = self._local_tokenizer.decode(
                outputs[0][input_length:], skip_special_tokens=True
            ).strip()
            return clean_conversational_response(answer)
        except Exception as exc:
            print(f"[LLMGenerator] Local generation error: {exc}")
            return ""

    def generate_local_causal_lm(
        self, query: str, context: str = "", scope: str = "the organization",
        model_id: str = "Qwen/Qwen2.5-1.5B-Instruct", max_new_tokens: int = 512,
    ) -> str:
        """Public alias for _call_local_causal_lm."""
        return self._call_local_causal_lm(query, context, scope, model_id, max_new_tokens)

    # ── Extractive QA engine (deepset/roberta-base-squad2) ────────────────────

    def _call_extractive_qa(
        self, query: str, context: str,
        model_id: str = "deepset/roberta-base-squad2",
    ) -> str:
        """
        Extractive QA using AutoModelForQuestionAnswering directly.
        Compatible with transformers v5+ (no 'question-answering' pipeline needed).
        Finds the best answer span in the provided context.
        """
        if self._qa_model_id != model_id or self._qa_pipeline is None:
            try:
                from transformers import AutoTokenizer, AutoModelForQuestionAnswering
                print(f"[LLMGenerator] Loading QA model: {model_id}")
                token_val = self._get_hf_token() or None
                tokenizer = AutoTokenizer.from_pretrained(model_id, token=token_val)
                model = AutoModelForQuestionAnswering.from_pretrained(model_id, token=token_val)
                model.eval()
                # Store tokenizer and model as a tuple in _qa_pipeline for reuse
                self._qa_pipeline = (tokenizer, model)
                self._qa_model_id = model_id
                print(f"[LLMGenerator] QA model loaded: {model_id}")
            except Exception as exc:
                print(f"[LLMGenerator] QA model load failed ({model_id}): {exc}")
                return ""

        if not context or not query:
            return ""

        try:
            import torch
            qa_tokenizer, qa_model = self._qa_pipeline

            # Split context into chunks to fit within model's max length
            context_chunks = [context[i:i+1500] for i in range(0, min(len(context), 4500), 1500)]
            best_answer = ""
            best_score = -1.0

            for ctx_chunk in context_chunks:
                if not ctx_chunk.strip():
                    continue
                inputs = qa_tokenizer(
                    query, ctx_chunk,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    return_offsets_mapping=True,
                )
                offset_mapping = inputs.pop("offset_mapping")[0]
                with torch.no_grad():
                    outputs = qa_model(**inputs)

                start_logits = outputs.start_logits[0]
                end_logits = outputs.end_logits[0]

                # Find best start/end positions using top-k candidates (efficient)
                input_ids = inputs["input_ids"][0]
                sep_idx = (input_ids == qa_tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
                ctx_start = int(sep_idx[0]) + 1 if len(sep_idx) > 0 else 1

                # Mask positions before context start
                start_logits[:ctx_start] = -1e10
                end_logits[:ctx_start] = -1e10

                n_best = min(20, len(start_logits) - ctx_start)
                if n_best <= 0:
                    continue
                top_starts = torch.topk(start_logits, n_best).indices
                top_ends = torch.topk(end_logits, n_best).indices
                start_probs = torch.softmax(start_logits, dim=-1)
                end_probs = torch.softmax(end_logits, dim=-1)

                # Vectorized search over start/end candidates
                s_scores = start_probs[top_starts]
                e_scores = end_probs[top_ends]
                score_mat = torch.outer(s_scores, e_scores)

                s_pos = top_starts.unsqueeze(1)
                e_pos = top_ends.unsqueeze(0)
                valid_mask = (e_pos >= s_pos) & ((e_pos - s_pos) <= 200)
                score_mat = score_mat * valid_mask.float()

                max_score = float(torch.max(score_mat))
                if max_score > best_score and max_score > 0.01:
                    flat_idx = int(torch.argmax(score_mat))
                    cur_best_start = int(top_starts[flat_idx // n_best])
                    cur_best_end = int(top_ends[flat_idx % n_best])
                    start_char = int(offset_mapping[cur_best_start][0])
                    end_char = int(offset_mapping[cur_best_end][1])
                    answer = ctx_chunk[start_char:end_char].strip()
                    if answer:
                        best_score = max_score
                        best_answer = answer

            if not best_answer or best_score < 0.01:
                return ""
            return clean_conversational_response(best_answer)
        except Exception as exc:
            print(f"[LLMGenerator] Extractive QA error: {exc}")
            return ""

    # ── Seq2Seq engine (google/flan-t5-base) ──────────────────────────────────

    def _call_seq2seq(
        self, query: str, context: str, scope: str,
        model_id: str = "google/flan-t5-base",
        max_new_tokens: int = 512,
    ) -> str:
        """
        Generate answer using AutoModelForSeq2SeqLM directly.
        Compatible with transformers v5+ (no 'text2text-generation' pipeline needed).
        Uses google/flan-t5-base for text-to-text generation.
        """
        if self._seq2seq_model_id != model_id or self._seq2seq_pipeline is None:
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                print(f"[LLMGenerator] Loading Seq2Seq model: {model_id}")
                token_val = self._get_hf_token() or None
                tokenizer = AutoTokenizer.from_pretrained(model_id, token=token_val)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id, token=token_val)
                model.eval()
                # Store tokenizer and model as a tuple
                self._seq2seq_pipeline = (tokenizer, model)
                self._seq2seq_model_id = model_id
                print(f"[LLMGenerator] Seq2Seq model loaded: {model_id}")
            except Exception as exc:
                print(f"[LLMGenerator] Seq2Seq model load failed ({model_id}): {exc}")
                return ""

        try:
            import torch
            s2s_tokenizer, s2s_model = self._seq2seq_pipeline

            # Flan-T5 works best with a friendly conversational instruction prefix
            prompt = (
                f"You are a friendly HR representative speaking directly to an employee.\n"
                f"Answer the employee's question based only on the context provided.\n"
                f"Match the user's requested format — table, list, comparison, or paragraph.\n"
                f"If the answer is not in the context, politely state that it was not found.\n\n"
                f"Context: {context[:2048]}\n\n"
                f"Question: {query}\n\n"
                f"Response:"
            )
            inputs = s2s_tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512,
            )
            with torch.no_grad():
                outputs = s2s_model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                )
            answer = s2s_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            return clean_conversational_response(answer) if answer else ""
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
            "max_tokens": 1024,
            "temperature": 0.0,
        }
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        endpoints = [
            "https://router.huggingface.co/v1/chat/completions",
            "https://router.huggingface.co/hf-inference/v1/chat/completions",
            f"https://router.huggingface.co/models/{model_id}/v1/chat/completions",
        ]
        
        # Extractive QA models (e.g. RoBERTa) cannot use Chat Completions API
        engine_type = HF_MODELS.get(model_id, {}).get("engine", "causal_lm")
        if engine_type == "extractive_qa":
            return ""

        # Try requested model, and fallback to serverless Qwen/Qwen2.5-72B-Instruct if requested 3B isn't hosted serverless
        model_candidates = [model_id, "Qwen/Qwen2.5-72B-Instruct"] if model_id != "Qwen/Qwen2.5-72B-Instruct" else [model_id]

        for cand_model in model_candidates:
            payload["model"] = cand_model
            for url in endpoints:
                try:
                    import requests
                    res = requests.post(url, json=payload, headers=headers, timeout=5)
                    if res.status_code == 200:
                        data = res.json()
                        if data.get("choices"):
                            ans = data["choices"][0]["message"]["content"].strip()
                            return clean_conversational_response(ans)
                    elif res.status_code == 403:
                        print("[LLMGenerator] HF 403: Token lacks 'Inference Providers' scope.")
                        return ""
                    else:
                        continue
                except requests.exceptions.Timeout:
                    print(f"[LLMGenerator] HF cloud timeout ({url}) — trying next endpoint")
                    continue
                except requests.exceptions.RequestException as exc:
                    print(f"[LLMGenerator] HF cloud notice ({url}): {exc}")
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
                    outputs = model.generate(**inputs, max_new_tokens=512)
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
        use_local_engine: bool = False,
        conversation_context: str = "",
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
        if CLOUD_ONLY_MODE and use_local_engine:
            print("[LLMGenerator] CLOUD_ONLY_MODE enabled — forcing cloud API (no local weights)")
            use_local_engine = False

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

        # Add conversation context for follow-up questions
        if conversation_context:
            formatted_context = conversation_context + "\n\nCurrent policy context:\n" + formatted_context

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
