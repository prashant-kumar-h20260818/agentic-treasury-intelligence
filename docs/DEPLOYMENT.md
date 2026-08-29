# Deployment Guide

## Local development

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

## Environment variables

Optional:

```env
GOOGLE_API_KEY=...
GOOGLE_MODEL=gemini-3.7-flash
BENCHMARK_TICKER=SPY
MAX_ASSETS=12
```

Without `GOOGLE_API_KEY`, all quantitative agents, charts, stress tests, allocation logic, governance and HITL still work. Only the executive brief uses fallback text.

## Streamlit Community Cloud

1. Push the repository to GitHub.
2. In Streamlit Community Cloud, create an app from the repository.
3. Main file: `app.py`.
4. Add `GOOGLE_API_KEY` to secrets if you want Gemini.
5. Deploy.

Do not commit API keys.

## Docker

```bash
docker build -t agentic-treasury-intelligence .
docker run --rm -p 8501:8501 --env-file .env agentic-treasury-intelligence
```

## CI

`.github/workflows/ci.yml`:

- installs Python 3.13
- installs dependencies
- compiles application source
- runs Pytest

## Production notes

The included `InMemorySaver` is suitable for a demonstration but not durable across server restarts. A bank/enterprise implementation should persist checkpoints externally and use authenticated user identities as part of the thread/session model.
