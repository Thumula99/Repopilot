from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow `streamlit run app.py` from the project root without installing the package.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import streamlit as st

from repopilot.chunker import chunk_documents
from repopilot.config import get_settings
from repopilot.evaluator import parse_eval_cases, run_retrieval_eval, summarize_eval
from repopilot.github_loader import GitHubLoader, parse_github_url
from repopilot.graph import RepoPilotAgent
from repopilot.store import ChromaStore, safe_collection_name
from repopilot.tickets import classify_priority


st.set_page_config(page_title="RepoPilot", page_icon="🛠️", layout="wide")


def get_secret_or_env(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)


st.title("🛠️ RepoPilot")
st.caption("A free-friendly Agentic RAG support engineer for GitHub repositories.")

with st.sidebar:
    st.header("Setup")
    provider = st.selectbox("LLM provider", ["offline", "gemini", "ollama"], index=0)
    gemini_key = st.text_input(
        "Gemini API key",
        value=get_secret_or_env("GEMINI_API_KEY", ""),
        type="password",
        help="Optional. Leave empty to use offline extractive mode.",
    )
    gemini_model = st.text_input("Gemini model", value=get_secret_or_env("GEMINI_MODEL", "gemini-2.5-flash"))
    ollama_model = st.text_input("Ollama model", value=get_secret_or_env("OLLAMA_MODEL", "llama3.1"))
    github_token = st.text_input(
        "GitHub token",
        value=get_secret_or_env("GITHUB_TOKEN", ""),
        type="password",
        help="Optional. Public repos work without it, but GitHub rate limits are lower.",
    )
    chroma_dir = st.text_input("Chroma directory", value=get_secret_or_env("CHROMA_DIR", "data/chroma"))

    st.divider()
    st.header("Repository")
    repo_url = st.text_input("GitHub repo URL", placeholder="https://github.com/owner/repo")
    max_files = st.slider("Max files to ingest", min_value=10, max_value=200, value=60, step=10)
    max_issues = st.slider("Max issues to ingest", min_value=0, max_value=100, value=30, step=10)

settings = get_settings(
    llm_provider=provider,
    gemini_api_key=gemini_key,
    gemini_model=gemini_model,
    ollama_model=ollama_model,
    github_token=github_token,
    chroma_dir=chroma_dir,
)

collection_name = "repopilot"
if repo_url.strip():
    try:
        collection_name = safe_collection_name(parse_github_url(repo_url).slug)
    except Exception:
        collection_name = "repopilot"

store = ChromaStore(settings.chroma_dir, collection_name=collection_name)
agent = RepoPilotAgent(store, settings)

col_a, col_b, col_c = st.columns(3)
col_a.metric("Indexed chunks", store.count())
col_b.metric("Collection", collection_name)
col_c.metric("Mode", provider)

if st.sidebar.button("Ingest / re-index repo", type="primary", disabled=not bool(repo_url.strip())):
    with st.spinner("Fetching GitHub repo, chunking files, and indexing locally..."):
        try:
            loader = GitHubLoader(token=github_token or None)
            documents = loader.load_repo(repo_url, max_files=max_files, max_issues=max_issues)
            chunks = chunk_documents(documents)
            store.reset()
            count = store.upsert(chunks)
            st.success(f"Indexed {count} chunks from {len(documents)} repo documents/issues.")
        except Exception as exc:
            st.error(f"Ingestion failed: {exc}")

ask_tab, debug_tab, issue_tab, eval_tab, about_tab = st.tabs(
    ["Ask repo", "Debug", "Issue draft", "Evaluation", "About"]
)

with ask_tab:
    st.subheader("Ask questions about the repository")
    question = st.text_area(
        "Question",
        placeholder="How does authentication work? Where is the login logic? How do I run this project locally?",
        height=100,
    )
    if st.button("Ask", disabled=not bool(question.strip())):
        with st.spinner("Searching repo context and generating answer..."):
            result = agent.run(question)
            st.markdown(result.get("answer", "No answer generated."))
            contexts = result.get("contexts", [])
            if contexts:
                with st.expander("Retrieved sources"):
                    for index, hit in enumerate(contexts, start=1):
                        score = f" — score {hit.score:.2f}" if hit.score is not None else ""
                        st.markdown(f"**Source {index}{score}:** {hit.label}")
                        if hit.metadata.get("url"):
                            st.link_button("Open source", hit.metadata["url"])
                        st.code(hit.text[:1200])

with debug_tab:
    st.subheader("Debug a problem using repo context")
    bug = st.text_area(
        "Problem / error message",
        placeholder="I get 401 when calling /api/profile after login. What should I check?",
        height=130,
    )
    if st.button("Debug", disabled=not bool(bug.strip())):
        with st.spinner("Finding related files/issues and preparing checklist..."):
            result = agent.run(bug)
            st.markdown(result.get("answer", "No debug answer generated."))
            priority = classify_priority(bug)
            st.info(f"Rule-based priority estimate: **{priority.priority}** — {priority.reason}")

with issue_tab:
    st.subheader("Generate a GitHub issue draft")
    problem = st.text_area(
        "Describe the bug or feature request",
        placeholder="The app crashes after successful login on mobile screens.",
        height=140,
    )
    if st.button("Create issue draft", disabled=not bool(problem.strip())):
        with st.spinner("Drafting issue..."):
            result = agent.run(f"Create GitHub issue / bug report: {problem}")
            st.markdown(result.get("answer", "No issue draft generated."))

with eval_tab:
    st.subheader("Simple retrieval evaluation")
    st.write(
        "Add test cases in the format `question | expected_keyword`. "
        "This checks whether the expected keyword appears in the top retrieved chunks."
    )
    raw_cases = st.text_area(
        "Evaluation cases",
        value="How do I run this project locally? | install\nWhere is authentication handled? | auth",
        height=150,
    )
    if st.button("Run retrieval evaluation"):
        cases = parse_eval_cases(raw_cases)
        if not cases:
            st.warning("No valid cases found. Use: question | expected_keyword")
        else:
            results = run_retrieval_eval(store, cases)
            summary = summarize_eval(results)
            c1, c2, c3 = st.columns(3)
            c1.metric("Cases", summary["total"])
            c2.metric("Passed", summary["passed"])
            c3.metric("Accuracy", f"{summary['accuracy']:.0%}")
            st.dataframe(
                pd.DataFrame([result.__dict__ for result in results]),
                use_container_width=True,
            )

with about_tab:
    st.subheader("What this project demonstrates")
    st.markdown(
        """
        **RepoPilot** demonstrates internship-relevant AI engineering skills:

        - Agent routing with LangGraph
        - Local RAG over GitHub files and issues
        - Persistent Chroma vector search
        - Free offline embeddings using scikit-learn HashingVectorizer
        - Optional Gemini or local Ollama LLM generation
        - Bug-report generation and priority classification
        - Retrieval evaluation dashboard

        Start in `offline` mode for zero cost. Add a free Gemini API key or use Ollama locally when you want stronger answers.
        """
    )
