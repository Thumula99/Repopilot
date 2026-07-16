from __future__ import annotations

import textwrap
from typing import Iterable

import requests

from .config import Settings
from .models import SearchHit


SYSTEM_STYLE = """You are RepoPilot, an AI support engineer for software repositories.
Be practical, concise, and honest. Use the provided context only when answering
repo-specific questions. If the context is insufficient, say what is missing.
Always include concrete next steps when debugging.
"""


def format_context(hits: Iterable[SearchHit], max_chars: int = 9000) -> str:
    blocks = []
    used = 0
    for idx, hit in enumerate(hits, start=1):
        url = hit.metadata.get("url", "")
        path = hit.metadata.get("path") or hit.metadata.get("title") or "unknown"
        header = f"[Source {idx}] {hit.metadata.get('source', 'source')} | {path} | {url}"
        text = hit.text.strip()
        block = f"{header}\n{text}"
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining <= 300:
                break
            block = block[:remaining]
        blocks.append(block)
        used += len(block)
    return "\n\n---\n\n".join(blocks)


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def generate(self, prompt: str, contexts: list[SearchHit] | None = None) -> str:
        provider = self.settings.llm_provider.lower()
        if provider == "gemini" and self.settings.gemini_api_key:
            return self._generate_gemini(prompt)
        if provider == "ollama":
            return self._generate_ollama(prompt)
        return self._generate_offline(prompt, contexts or [])

    def _generate_gemini(self, prompt: str) -> str:
        try:
            from google import genai

            client = genai.Client(api_key=self.settings.gemini_api_key)
            response = client.models.generate_content(
                model=self.settings.gemini_model,
                contents=prompt,
            )
            return (getattr(response, "text", None) or "").strip() or "Gemini returned an empty response."
        except Exception as exc:
            return f"Gemini call failed, so I could not generate an LLM answer. Error: {exc}"

    def _generate_ollama(self, prompt: str) -> str:
        try:
            response = requests.post(
                f"{self.settings.ollama_base_url.rstrip('/')}/api/generate",
                json={"model": self.settings.ollama_model, "prompt": prompt, "stream": False},
                timeout=90,
            )
            response.raise_for_status()
            return response.json().get("response", "").strip() or "Ollama returned an empty response."
        except Exception as exc:
            return f"Ollama call failed. Is Ollama running locally? Error: {exc}"

    def _generate_offline(self, prompt: str, contexts: list[SearchHit]) -> str:
        """Free fallback. It does not hallucinate; it summarizes retrieved chunks."""
        if not contexts:
            return (
                "I do not have enough indexed repo context yet. Ingest a GitHub repository first, "
                "then ask again."
            )

        bullets = []
        for hit in contexts[:4]:
            path = hit.metadata.get("path") or hit.metadata.get("title") or "source"
            snippet = " ".join(hit.text.strip().split())[:500]
            bullets.append(f"- **{path}**: {snippet}...")

        return (
            "I am running in **offline mode**, so this is an extractive answer from the retrieved sources.\n\n"
            "Most relevant repo evidence:\n"
            + "\n".join(bullets)
            + "\n\nFor better natural-language reasoning, set `LLM_PROVIDER=gemini` with a free Gemini API key "
            "or use `LLM_PROVIDER=ollama` with a local model."
        )


def build_answer_prompt(question: str, contexts: list[SearchHit]) -> str:
    return textwrap.dedent(
        f"""
        {SYSTEM_STYLE}

        Task: Answer the user's repository question using the context.

        User question:
        {question}

        Context:
        {format_context(contexts)}

        Answer format:
        1. Direct answer
        2. Evidence from repo sources
        3. Suggested next steps
        """
    ).strip()


def build_debug_prompt(question: str, contexts: list[SearchHit]) -> str:
    return textwrap.dedent(
        f"""
        {SYSTEM_STYLE}

        Task: Act as a junior support/debugging engineer. Diagnose the likely issue using only the context.

        User problem:
        {question}

        Context:
        {format_context(contexts)}

        Answer format:
        - Likely cause
        - Files/issues to inspect
        - Step-by-step debugging checklist
        - What extra information is needed if uncertain
        """
    ).strip()


def build_issue_prompt(problem: str, contexts: list[SearchHit]) -> str:
    return textwrap.dedent(
        f"""
        {SYSTEM_STYLE}

        Task: Create a high-quality GitHub issue draft based on the user's problem and repo context.

        User problem:
        {problem}

        Context:
        {format_context(contexts)}

        Output markdown with these sections:
        Title:
        Summary:
        Steps to reproduce:
        Expected behavior:
        Actual behavior:
        Possible cause:
        Suggested labels:
        Related files/issues:
        """
    ).strip()
