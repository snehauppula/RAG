# RAG PDF Assistant

Retrieval-Augmented Generation over your PDFs: ingest, embed, store in Chroma, answer with Groq.

## Features

- Modular Python package under `project/`
- Jupyter notebooks in `notebook/` (original walkthrough)
- CLI: ingest, test suite, single-question `--ask`
- Streamlit UI for demos
- Source citations and retrieval confidence

## Quick start

```bash
cd RAG
python -m venv .venv
.venv\Scripts\activate
pip install -r project/requirements.txt
copy .env.example .env
# Edit .env: set GROQ_API_KEY

python -m project --reset-index --ingest-only
python -m project --test --test-llm
streamlit run app.py
```

## Project layout

| Path | Purpose |
|------|---------|
| `project/` | RAG modules (`config`, `data_loader`, `embedding`, `vector_store`, `query`, `pipeline`) |
| `app.py` | Streamlit UI |
| `notebook/` | Jupyter notebooks (original pipeline + experiments) |
| `data/pdf/` | Your PDF corpus |
| `tests/` | pytest unit + integration tests |
| `TESTING.md` | How to verify the pipeline |

## Files to push to GitHub (include notebook)

```
README.md
.gitignore
.env.example
app.py
TESTING.md
project/              # all Python modules + requirements.txt + README
notebook/
  data_loader.ipynb   # main notebook (refactored into project/)
  document.ipynb
tests/
data/pdf/             # optional: sample PDFs
```

**Do not push:** `.env`, `.venv/`, `data/vector_store/`, `__pycache__/`, `src/` (old unused code)

**Notebook:** commit the `.ipynb` files; outputs are cleared (no API keys in the file). Re-run cells after clone.

## Docs

- [project/README.md](project/README.md) - module details
- [TESTING.md](TESTING.md) - test checklist

## Requirements

- Python 3.10+
- Groq API key (free tier works for demos)
- PDFs in `data/pdf/`

## License

Educational / portfolio use. PDF sample papers are for local RAG practice.
