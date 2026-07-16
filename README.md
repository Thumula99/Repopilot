# RepoPilot — Agentic RAG Support Engineer

RepoPilot is a free-friendly AI project for students who want to show internship-ready AI engineering skills. It connects to a public GitHub repository, indexes files and issues, answers repo questions, helps debug errors, drafts GitHub issues, classifies ticket priority, and includes a simple retrieval evaluation dashboard.

## Why this project is strong for AI internships

Most student projects stop at "chat with PDF". RepoPilot shows a more practical production workflow:

- GitHub repository ingestion
- RAG over code, docs, and issues
- LangGraph agent routing
- Local Chroma vector search
- Free offline embeddings
- Optional Gemini or local Ollama generation
- Bug report generation
- Ticket priority classification
- Basic evaluation dashboard
- Streamlit UI

## Features

### 1. Ask repo questions

Example prompts:

```text
How do I run this project locally?
Where is authentication handled?
Explain the architecture of this repo.
Which files should I read first?
```

### 2. Debug problems

Example prompts:

```text
I get 401 when calling /api/profile after login. What should I check?
The app crashes after successful login on mobile screens.
```

RepoPilot searches relevant files and issues, then returns a likely cause and debugging checklist.

### 3. Draft GitHub issues

RepoPilot generates structured issue drafts with:

- Title
- Summary
- Steps to reproduce
- Expected behavior
- Actual behavior
- Possible cause
- Suggested labels
- Related files/issues

### 4. Priority classification

A free rule-based classifier estimates issue priority as Low, Medium, High, or Critical. You can later replace this with a small ML model.

### 5. Retrieval evaluation

Add test cases like:

```text
How do I run this project locally? | install
Where is authentication handled? | auth
```

The app checks whether expected keywords appear in the top retrieved chunks.

## Architecture

```text
Streamlit UI
   ↓
RepoPilot Agent - LangGraph
   ↓
Router node
   ├── Ask repo → Retriever → Answer generator
   ├── Debug → Retriever → Debug checklist
   ├── Issue draft → Retriever → Issue generator
   └── Priority → Rule classifier
   ↓
Chroma vector store
   ↓
GitHub files + GitHub issues
```

## Free-first tech stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Agent workflow | LangGraph |
| Vector store | Chroma local persistent client |
| Embeddings | scikit-learn HashingVectorizer |
| LLM option 1 | Offline extractive mode, zero API cost |
| LLM option 2 | Gemini API free tier where available |
| LLM option 3 | Ollama local model |
| Repo source | Public GitHub API |

## Quick start

### 1. Clone or unzip this project

```bash
cd repopilot
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows PowerShell
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

### 5. Ingest a repo

Paste a public GitHub repository URL, for example:

```text
https://github.com/streamlit/streamlit
```

Then click **Ingest / re-index repo**.

## Using Gemini

1. Create a Gemini API key in Google AI Studio.
2. Copy `.env.example` to `.env`.
3. Set:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
```

You can also paste the key in the Streamlit sidebar.

## Using Ollama locally

Install Ollama, then run:

```bash
ollama pull llama3.1
ollama serve
```

Set the sidebar provider to `ollama`.

## Optional GitHub token

Public repos work without a token, but GitHub rate limits unauthenticated requests. Add a token if you index many repos.

```env
GITHUB_TOKEN=your_optional_token
```

## Suggested portfolio demo script

Record a 2–3 minute video:

1. Show the architecture diagram from this README.
2. Ingest a public repo.
3. Ask: "How do I run this project locally?"
4. Ask: "Where is authentication handled?"
5. Debug a fake error.
6. Generate a GitHub issue draft.
7. Show the evaluation tab.

## Resume bullet

Built RepoPilot, an agentic RAG support engineer using LangGraph, Streamlit, Chroma, and GitHub API. The app indexes repository files and issues, answers repo questions with retrieved sources, generates bug reports, classifies ticket priority, and includes a retrieval evaluation dashboard.

## Roadmap

Good upgrades after the MVP:

- Replace HashingVectorizer with a stronger local embedding model.
- Add GitHub issue creation through OAuth or a personal access token.
- Add repo architecture visualization.
- Add source-code dependency graph.
- Train a small ticket-priority classifier using public issue datasets.
- Add Docker deployment on Hugging Face Spaces.
- Add CI tests with GitHub Actions.
