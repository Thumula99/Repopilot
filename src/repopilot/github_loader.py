from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import Iterable, List
from urllib.parse import urlparse

import requests

from .models import SourceDocument
from .chunker import stable_id


CODE_EXTENSIONS = {
    ".md",
    ".mdx",
    ".txt",
    ".rst",
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".php",
    ".rb",
    ".cs",
    ".rs",
    ".html",
    ".css",
    ".scss",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".env.example",
}

SKIP_PATH_PARTS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    ".next",
    "coverage",
    "__pycache__",
    ".pytest_cache",
    "target",
}

MAX_FILE_BYTES = 180_000


@dataclass
class RepoRef:
    owner: str
    repo: str

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.repo}"


def parse_github_url(value: str) -> RepoRef:
    """Parse owner/repo from a GitHub URL or owner/repo string."""
    value = value.strip().removesuffix(".git")
    if re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", value):
        owner, repo = value.split("/", 1)
        return RepoRef(owner=owner, repo=repo)

    parsed = urlparse(value)
    if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
        raise ValueError("Please enter a GitHub repo URL like https://github.com/owner/repo")
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise ValueError("GitHub URL must include owner and repository name")
    return RepoRef(owner=parts[0], repo=parts[1])


class GitHubLoader:
    def __init__(self, token: str | None = None, timeout: int = 20):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/vnd.github+json"})
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})

    def _get_json(self, url: str, params: dict | None = None):
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_default_branch(self, repo: RepoRef) -> str:
        data = self._get_json(f"https://api.github.com/repos/{repo.owner}/{repo.repo}")
        return data.get("default_branch") or "main"

    def list_files(self, repo: RepoRef, branch: str, max_files: int) -> list[dict]:
        url = f"https://api.github.com/repos/{repo.owner}/{repo.repo}/git/trees/{branch}"
        data = self._get_json(url, params={"recursive": "1"})
        files = []
        for item in data.get("tree", []):
            if item.get("type") != "blob":
                continue
            path = item.get("path", "")
            if not self._should_include_file(path, item.get("size", 0)):
                continue
            files.append(item)
            if len(files) >= max_files:
                break
        return files

    def _should_include_file(self, path: str, size: int) -> bool:
        lower_path = path.lower()
        if size and size > MAX_FILE_BYTES:
            return False
        if any(part in SKIP_PATH_PARTS for part in lower_path.split("/")):
            return False
        if lower_path.endswith(".env.example"):
            return True
        return any(lower_path.endswith(ext) for ext in CODE_EXTENSIONS)

    def fetch_file_text(self, repo: RepoRef, branch: str, path: str) -> str:
        url = f"https://api.github.com/repos/{repo.owner}/{repo.repo}/contents/{path}"
        data = self._get_json(url, params={"ref": branch})
        content = data.get("content")
        encoding = data.get("encoding")
        if not content:
            return ""
        if encoding == "base64":
            raw = base64.b64decode(content).decode("utf-8", errors="ignore")
            return raw
        return str(content)

    def load_repo_files(self, repo_url: str, max_files: int = 80) -> list[SourceDocument]:
        repo = parse_github_url(repo_url)
        branch = self.get_default_branch(repo)
        docs: list[SourceDocument] = []
        for item in self.list_files(repo, branch, max_files=max_files):
            path = item.get("path", "")
            try:
                text = self.fetch_file_text(repo, branch, path)
            except requests.HTTPError:
                continue
            if not text.strip():
                continue
            docs.append(
                SourceDocument(
                    id=stable_id(repo.slug, branch, path),
                    text=f"File: {path}\nRepository: {repo.slug}\n\n{text}",
                    metadata={
                        "source": "github_file",
                        "repo": repo.slug,
                        "branch": branch,
                        "path": path,
                        "url": f"https://github.com/{repo.slug}/blob/{branch}/{path}",
                    },
                )
            )
        return docs

    def load_issues(self, repo_url: str, max_issues: int = 50) -> list[SourceDocument]:
        repo = parse_github_url(repo_url)
        docs: list[SourceDocument] = []
        page = 1
        while len(docs) < max_issues:
            url = f"https://api.github.com/repos/{repo.owner}/{repo.repo}/issues"
            issues = self._get_json(
                url,
                params={"state": "all", "per_page": min(100, max_issues), "page": page},
            )
            if not issues:
                break
            for issue in issues:
                # Pull requests appear in the issues endpoint; skip them for this app.
                if "pull_request" in issue:
                    continue
                number = issue.get("number")
                title = issue.get("title", "Untitled issue")
                body = issue.get("body") or ""
                labels = ", ".join(label.get("name", "") for label in issue.get("labels", []))
                text = (
                    f"GitHub issue #{number}: {title}\n"
                    f"Repository: {repo.slug}\n"
                    f"State: {issue.get('state')}\n"
                    f"Labels: {labels}\n"
                    f"URL: {issue.get('html_url')}\n\n"
                    f"{body}"
                )
                docs.append(
                    SourceDocument(
                        id=stable_id(repo.slug, "issue", str(number)),
                        text=text,
                        metadata={
                            "source": "github_issue",
                            "repo": repo.slug,
                            "issue_number": number,
                            "title": title,
                            "state": issue.get("state"),
                            "labels": labels,
                            "url": issue.get("html_url"),
                        },
                    )
                )
                if len(docs) >= max_issues:
                    break
            page += 1
        return docs

    def load_repo(self, repo_url: str, max_files: int = 80, max_issues: int = 50) -> list[SourceDocument]:
        documents = self.load_repo_files(repo_url, max_files=max_files)
        try:
            documents.extend(self.load_issues(repo_url, max_issues=max_issues))
        except requests.HTTPError:
            # Issues may be disabled or rate-limited; files are still useful.
            pass
        return documents
