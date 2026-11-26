import httpx2

GITHUB_API_URL = "https://api.github.com"
GITHUB_API_VERSION = "2022-11-28"
USER_AGENT = "openmfp-copilot"


class ReadmeNotFoundError(Exception):
    def __init__(self, owner: str, repo: str) -> None:
        super().__init__(f"Repository '{owner}/{repo}' has no README or does not exist")
        self.owner = owner
        self.repo = repo


class GitHubClient:
    def __init__(self, http: httpx2.AsyncClient) -> None:
        self._http = http

    async def get_readme(self, owner: str, repo: str) -> str:
        response = await self._http.get(
            f"{GITHUB_API_URL}/repos/{owner}/{repo}/readme",
            headers={
                "Accept": "application/vnd.github.raw+json",
                "X-GitHub-Api-Version": GITHUB_API_VERSION,
                "User-Agent": USER_AGENT,
            },
        )
        if response.status_code == httpx2.codes.NOT_FOUND:
            raise ReadmeNotFoundError(owner, repo)
        response.raise_for_status()
        return response.text
