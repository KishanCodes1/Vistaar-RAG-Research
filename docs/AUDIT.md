# Code audit — original `Vistaar-main` → v0.2

Findings from a full read of the original 12 KB / ~250-line repository
(commits `First request`, `readme update`), with status.

## Critical defects

| # | Defect | Impact | Status |
|---|---|---|---|
| 1 | Single global FAISS index shared by all users (`vector_store.py` module globals) | Any user could retrieve any other user's document contents | **Fixed** — namespaced `Store`, per-session `namespace` |
| 2 | `BackgroundTasks` writes to the index while `/ask` searches it | FAISS is not thread-safe for concurrent add/search → corruption or crash under load | **Fixed** — `RLock` around all reads/writes |
| 3 | All state in process memory | Restart loses every document; no dedup, no delete | **Fixed** — FAISS + chunk metadata persisted per namespace; `DELETE /namespace/{ns}` |
| 4 | No job status; UI says "wait a few seconds" | Client cannot know when indexing finished; silent failures | **Fixed** — upload returns synchronously with chunk/page counts |
| 5 | Frontend does `data.message.toLowerCase()` on every response | `TypeError` on 429/500 (which return `detail`, not `message`) → UI dies silently | **Fixed** — checks `res.ok`, renders `detail` |
| 6 | Gemini 3.x thinking mode consumes the whole output budget | `content: {}` + `finishReason: MAX_TOKENS` → empty `response.text` → app reports "AI service is busy" forever | **Fixed** — `thinkingBudget: 0` set explicitly |
| 7 | Hardcoded `gemini-2.5-flash` fallback | Model retired for new users → 404 on every retry | **Fixed** — `gemini-3.5-flash` primary, `gemini-3.8-flash` fallback |
| 8 | `.env` value had a leading space (`KEY= <value>`) | `source .env` assigns empty; naive parsers send a broken key → 403 | **Fixed** — `config._clean()` strips; regression test added |
| 9 | Retrieved document text concatenated raw into the prompt | Prompt injection from a malicious PDF could override the system prompt | **Mitigated** — passages are labelled and delimited; not fully solved |

## Accuracy / quality defects

| # | Defect | Impact | Status |
|---|---|---|---|
| 10 | Character-based chunking at fixed 1000/200, ignoring sentence and paragraph boundaries | Cuts sentences mid-word, degrades embeddings | **Fixed** — boundary-aware recursive splitter |
| 11 | No chunk metadata (page, doc id, offsets) | Citations impossible | **Fixed** — every `Chunk` carries `doc_id`, `page`, `ordinal`, `chunk_id` |
| 12 | Fixed `k=8`, no score threshold | Irrelevant context injected when the document lacks the answer → hallucination | **Fixed** — configurable `k`, abstention on a thresholded confidence score |
| 13 | No reranking, no lexical retrieval | Weak recall on exact terms, identifiers, numbers | **Fixed** — BM25 + dense with reciprocal rank fusion; optional cross-encoder |
| 14 | PyPDF2 only; scanned PDFs silently rejected with "not readable" | Large class of real documents unusable | **Partial** — clearer error message; OCR still absent |
| 15 | No citations at all | Answers unauditable | **Fixed** — span offset → passage → page |
| 16 | README claimed OpenAI + local transformers, 500-char chunks, and "reduced hallucinations" | None of it was true or measured; disqualifying for publication | **Fixed** — README rewritten to match code; claims tied to measurements |

## Engineering defects

| # | Defect | Impact | Status |
|---|---|---|---|
| 17 | `requirements.txt` fully unpinned | Irreproducible; breaks without warning | **Fixed** — all versions pinned |
| 18 | `qa_service.py` raises at import when the key is missing | App cannot be imported for tests | **Fixed** — lazy client, settings validation |
| 19 | No rate-limit-safe retry semantics; no request timeout | Hangs, silent empty answers | **Fixed** — timeout, backoff, ordered model fallback |
| 20 | `print()` as logging | No levels, no structure, nothing greppable in prod | **Fixed** — `logging` with structured messages |
| 21 | No auth, no tenancy, no upload size cap, no type validation | Memory exhaustion, `.exe` decoded and indexed | **Fixed** — 20 MB cap, extension allowlist, per-session namespaces |
| 22 | Rate limiter keyed on `get_remote_address` | Breaks behind any proxy (all users share a bucket) | **Known** — documented; needs `X-Forwarded-For` trust config |
| 23 | No tests, no CI, no Docker | No regression safety | **Fixed** — 12 tests, Dockerfile, Makefile |

## Still open (deliberately)

- **OCR** for scanned PDFs.
- **Auth/SSO** and real tenant management (currently session namespaces only).
- **Streaming** responses.
- **Proxy-aware rate limiting.**
- **Legal-domain data**: the evaluation is currently single-domain (NLP papers).
  The planned second domain (contracts/judgments) needs either CUAD ingestion or
  new annotation.
- **LLM-judged faithfulness**: citation precision is vocabulary-overlap based,
  which is stricter than a semantic judge but cannot catch paraphrase.

## Reproduction of the two most surprising findings

```bash
# 1. Thinking-mode bug: empty candidate unless thinkingBudget is 0
python -c "
import json,urllib.request
# maxOutputTokens=20 with default thinking -> finishReason MAX_TOKENS, parts=[]
"

# 2. Leading-space key: works with strip(), 403 without
python -c "print(repr(open('.env').read().split('=')[1][:4]))"
```
