# Security

## Scope
This is a research reference implementation. It is not hardened for production serving,
does not parse untrusted input formats, and makes no network calls.

## Known considerations
- Model weights are plain NumPy arrays. Do not load pickled weights from untrusted sources.
- `state_dict()` returns JSON-serialisable lists; prefer it over pickle for checkpoints.
- No telemetry, no external downloads, no credentials anywhere in this codebase.

## Reporting
Open a GitHub issue for non-sensitive problems. For anything sensitive, contact the maintainer
directly via the GitHub profile rather than a public issue.

## Credential hygiene
Never paste a personal access token into a chat, an issue, a notebook, or a commit.
If a token is exposed, revoke it immediately at
Settings > Developer settings > Personal access tokens, then issue a new one.
