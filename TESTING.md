# How to Test Your RAG Project

Run all commands from **`d:\RAG`** with the venv active:

```powershell
cd d:\RAG
.venv\Scripts\activate
```

---

## Level 1: One-command health check (start here)

```powershell
python -m project --test
```

**What it checks**

| Check | Meaning |
|-------|---------|
| `data_directory` | `data/pdf` exists |
| `vector_index_nonempty` | Chroma has chunks |
| `load_documents` / `split_documents` | Ingestion logic works |
| `retrieve_transformer_paper` | "attention" query hits Transformers.pdf |
| `retrieve_claude_paper` | Claude query hits Claude PDF |
| `retrieve_out_of_corpus_low_score` | Nonsense topic scores low |
| `rag_answer_groq` | Skipped unless you pass `--test-llm` |

**If index is empty**, rebuild then test:

```powershell
python -m project --reset-index --ingest-only
python -m project --test
```

**Exit code 0** = all checks passed.

---

## Level 2: Full RAG test (needs Groq API key)

```powershell
python -m project --test --test-llm
```

Also verifies Groq returns an answer about the Transformer (grounded in your PDFs).

Requires `GROQ_API_KEY` in `.env` at repo root.

---

## Level 3: Manual CLI tests

### Retrieval only (free, no API)

```powershell
python -m project --retrieve-only
```

### Ask one question

```powershell
python -m project --ask "What is attention is all you need?"
```

**Good answer:** mentions Transformer, self-attention, encoder/decoder.  
**Good sources:** `Transformers.pdf` with score > 0.4.

### Bad / edge case

```powershell
python -m project --ask "Hard Negative Mining Techniques"
```

**Expected:** weak confidence or "no relevant context" (topic not in your PDFs).

---

## Level 4: Automated pytest

```powershell
pip install pytest
pytest tests/test_rag_unit.py -v
pytest tests/test_rag_integration.py -v
```

| File | Speed | Needs |
|------|-------|-------|
| `test_rag_unit.py` | Fast | Nothing |
| `test_rag_integration.py` | Slower | Indexed data (`--ingest-only` first) |

---

## Level 5: Web UI

```powershell
pip install streamlit
streamlit run app.py
```

1. Confirm sidebar shows chunk count and **Groq API key loaded**.
2. Ask: `What is attention is all you need?`
3. Check answer + sources point to `Transformers.pdf`.
4. Use **Preview chunks only** to test retrieval without LLM.

---

## Level 6: Python API smoke test

```python
from project import RAGPipeline

rag = RAGPipeline.create()
print(rag.health())

hits = rag.retrieve("What is Claude used for?", top_k=3)
print(hits[0]["similarity_score"], hits[0]["metadata"]["source"])

result = rag.ask("What is the Transformer?")
rag.print_answer(result)
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `vector_index_nonempty` FAIL | `python -m project --reset-index --ingest-only` |
| `data_directory` FAIL | Add PDFs under `data/pdf/` |
| `rag_answer_groq` FAIL | Set `GROQ_API_KEY` in `.env` |
| Wrong answers | Lower `--min-score` or rephrase question |
| Duplicate chunks (305+) | `python -m project --reset-index --ingest-only` |

---

## Recommended test order (before a demo)

1. `python -m project --reset-index --ingest-only`
2. `python -m project --test`
3. `python -m project --ask "What is attention is all you need?"`
4. `python -m project --test --test-llm`
