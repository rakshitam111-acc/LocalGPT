---
title: LocalGPT ChatGPT AI Platform
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# LocalGPT — Self-Hosted ChatGPT Platform (No External APIs Needed)

A self-hosted, multi-user ChatGPT-style AI platform running open-source models (**Qwen 2.5 1.5B**, **Llama 3.2**) with in-memory PyTorch / Ollama execution, semantic document RAG, and multi-user authentication.

## 🚀 Cloud Deployment to Hugging Face Spaces (Free)

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Choose **Docker** as the Space SDK.
3. Push or upload the files from this repository to your Space.
4. Hugging Face will automatically build the Docker container and give you a free, public URL (e.g. `https://your-space.hf.space`) accessible to anyone in the world!

---

## 💻 Local Quick Start

Double-click `run_phase3.bat` or run:

```bash
uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.
