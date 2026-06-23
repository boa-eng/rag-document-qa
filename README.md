---
title: Crux Backend
emoji: 📄
colorFrom: red
colorTo: pink
sdk: docker
pinned: false
app_port: 7860
---

# Crux — Backend

FastAPI + sentence-transformers + Groq. Accepts document uploads, builds a FAISS + BM25 hybrid index per session, and streams answers via SSE.

## Environment variables (set as HF Secrets)

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Your Groq API key |
| `CRUX_ALLOWED_ORIGINS` | Yes | Comma-separated list of allowed frontend origins (e.g. `https://yoursite.netlify.app`) |
| `CRUX_PERSONA` | No | Optional system-prompt persona for white-label deployments |

## Local dev

```bash
cp .env.example .env   # add GROQ_API_KEY
pip install -r requirements.txt
uvicorn src.api:app --reload
```
