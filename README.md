# AI Makerspace · MLOps World 2026 · Retrieval workshop

Three 15-minute [marimo](https://marimo.io) notebooks, each one diagram-then-code
three times over. All three follow Nadia, a second-line support engineer at a
regional broadband operator, building the assistant that finds the answer a
ticket needs.

| Notebook | What it builds | The one thing it shows |
| --- | --- | --- |
| `notebooks/01_Agentic_RAG.py` | one agent, one tool: a BM25 chunk search it can call again | ranked chunks have a cap, and a question with the wrong words has to be reworded |
| `notebooks/02_LLM_Wiki_DCI.py` | the same agent with list / grep / read over the same pages | an index somebody maintains lets the agent go straight to the page, by name |
| `notebooks/03_Multimodal_Retrieval.py` | memos and charts in one Qdrant index, fused by rank, read by a vision model | find the chart with retrieval, read the number from the pixels |

01 and 02 ask the **same question** of the **same seven pages** with the same
model, system prompt, loop and budget. Only the tools change, so the two
traces compare cleanly on the slides.

## Setup

1. Install [uv](https://docs.astral.sh/uv/) and Python 3.12+, then from this
   folder:

   ```bash
   uv sync
   ```

2. Copy `.env.template` to `.env` and paste your OpenAI key. The defaults use
   `gpt-4.1-mini` for both the agents and the chart reading.

3. Open a notebook:

   ```bash
   uv run marimo edit notebooks/01_Agentic_RAG.py
   ```

   The first time you open `03_Multimodal_Retrieval.py` it downloads two
   embedding models (bge-small and CLIP, about 650 MB) into
   `~/.cache/fastembed`. Everything else is API calls.

Each notebook has one ▶️ button that makes the model calls. Press it before
going on stage; nothing else in the notebook needs a network call to render.

## On stage

Per notebook: three pairs of (diagram, the code it draws), about five minutes
a pair, 7 minutes of diagrams and 8 of code in all. The minute-by-minute
scripts are in `instructor/`.

## Layout

```
notebooks/            the three notebooks and their data
  data/corpus/        Nadia's seven pages (01 and 02)
  data/wiki/index.md  the page that makes them a wiki (02)
  data/multimodal/    six memos, six charts, the answer key (03)
  data/diagrams/      nine SVGs, three per notebook, embedded by helpers.ui.diagram
helpers/              nb.bootstrap() reads .env; ui.table / ui.diagram render output
instructor/           what to say, minute by minute
```

The notebooks and corpus are adapted from Week 3 of the
[AI Forward Deployed Engineer Certification](https://github.com/AI-Maker-Space/The-AI-Forward-Deployed-Engineer-Certification).
The multimodal corpus is fictional and machine-generated; see
`notebooks/data/multimodal/README.md`.

## Jupyter mirrors

To regenerate the `.ipynb` copies for people who prefer Jupyter:

```bash
uv run marimo export ipynb notebooks/01_Agentic_RAG.py -o notebooks/01_Agentic_RAG.ipynb
```

(and the same for 02 and 03). The `.py` files are the source of truth.
