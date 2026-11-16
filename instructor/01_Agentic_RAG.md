# 01 · Agentic RAG — the 15-minute script

Slot: 1:25–1:40. Everything pre-run (press ▶️ before you go up). Three pairs:
diagram, then the code it draws. ~2:20 on each diagram, ~2:40 on each code
cell. **Hard stop at 15.** If you run over, the cuts are at the bottom.

The sentence the whole notebook serves: *same model, same prompt, same loop,
same question as the next notebook — only the tool changes.* Say it at 0:00
and again at 14:30.

## Pair 1 · Setup (0:00–5:00)

**Diagram — "Seven pages, and a question that shares no words with its answer" (0:00–2:20)**

- Left column: seven files, one `Document` each, keyed by name. Point at
  `tariffs_and_slas`.
- Middle: the question, then the answer underneath it — section 3 of the
  tariffs page, £45, 24 months. *"Read the two boxes. 'Forty-five pounds' is
  not '£45'. 'Tied in' is not 'minimum term'. There is no word in common.
  That is deliberate: a word search has nothing to rank on."*
- Right: notebook 01 gives the agent **one tool**, `search_chunks`. *"It
  never sees a page name. It never sees an index."*
- Bottom strip: what is fixed for both notebooks. *"Hold onto this list;
  notebook 02 changes nothing on it."*

**Code (2:20–5:00)** — two cells.

- First cell: `PAGES` is the left column, a dict of name → text. `CFG` reads
  `.env`; the model is configuration. `QUESTION` is the box in the middle.
- Second cell: the right side of the strip. `ToolCallingLiteLLM` — one
  sentence: *"LlamaIndex only trusts models on its own list to call tools;
  this says 'trust this one'."* `SYSTEM` — read it aloud, it is three
  sentences. `DOCS` — the pages as LlamaIndex documents;
  `excluded_embed_metadata_keys` keeps the page name for citing and out of
  the index. *"Otherwise every section of the tariffs page matches the word
  'tariffs'."*

## Pair 2 · Task 1 (5:00–10:00)

**Diagram — "The cap cuts one section, and the answer sits past it" (5:00–7:20)**

- Left column, top to bottom: pages → `MarkdownNodeParser` (a section at
  every heading) → `BM25Retriever` (top 8) → `search_chunks` (top k, each cut
  at 1,000 characters) → the agent sees ranked sections only.
- Right: the 67 sections by length. The dashed line at 1,000 is the cut. One
  bar crosses it, and the answer to a different question sits at character
  1,445. *"Every chunk retriever has a cap somewhere — in the chunker, in
  the top-k, in the context window. This one is in the function, so you can
  see it."*

**Code (7:20–10:00)** — two cells.

- The four lines: `nodes`, `bm25`, `search_chunks`. Match them to the four
  boxes. The `[:1000]` is the dashed line.
- First table: 67 sections from 7 pages. Second table: `bm25.retrieve('LOS
  0x21')` — a query that *does* share words with its answer, so the
  glossary comes top. *"That is BM25 working."*
- Then the grey box: `search_chunks` called by hand with **our** question.
  Read the section headings in the result. *"It reaches the tariffs page —
  'month' and 'plan' got it there — but the wrong sections: service targets
  and the page intro. Nothing here says Fibre 500. The agent will have to
  reword."*

## Pair 3 · Task 2 (10:00–15:00)

**Diagram — "The loop runs until the model answers or the budget runs out" (10:00–12:20)**

- Grey frame = `run_agent`. Inside: `agent.run(question, max_iterations=6)`
  → the model → a tool call → the tool → one string back → the model again.
- Two exits: green, the model answered without asking for a tool; orange,
  six model calls spent, `WorkflowRuntimeError`. *"The budget is the only
  thing standing between you and an unbounded bill."*
- Right panel, THE TRACE: one row per `ToolCallResult` — tool, arguments,
  characters back. Two `search_chunks` rows; the second is a reworded query.
  Then what `run_agent` returns: answer, trace, evidence chars, latency,
  stopped.

**Code (12:20–15:00)** — three cells.

- `FunctionTool.from_defaults(search_chunks)` — the *tool* box; the JSON
  under the cell is literally what the model is handed: name, description
  from the docstring, `query` and `k` from the type hints.
- `run_agent` — the grey frame. Point at `stream_events`, `ToolCallResult`,
  the `except WorkflowRuntimeError`. Do not read the function; say what each
  of the five returned fields is.
- The run: the trace table, then the answer. *"Read the trace first. Query
  one is the model's rewording. If there is a query two, the first ranking
  didn't carry the tariff. It found it — and it never knew the page was
  called tariffs until a chunk said so."*
- Last cell, the one-row table: tool calls, evidence chars, latency,
  stopped. *"Four numbers. Notebook 02 produces the same four for the same
  question. Greg puts them side by side."*

## If you run over

Cut, in this order: the `bm25.retrieve('LOS 0x21')` table (say "BM25 works
when the words match" and move on); the JSON schema under Task 2 (say "the
docstring becomes the description"); the second Setup cell (say "model,
prompt, documents — fixed").

Never cut: the no-words-in-common point on the Setup diagram, the 1,000
dashed line, the trace table.
