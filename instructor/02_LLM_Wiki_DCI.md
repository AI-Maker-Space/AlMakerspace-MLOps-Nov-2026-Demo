# 02 · LLM Wiki / DCI — the 15-minute script

Slot: 1:50–2:05, straight after Greg's "LLM Wiki & DCI" intro. Pre-run.
Three pairs again; ~2:20 per diagram, ~2:40 per code cell. **Hard stop at 15**
— the break follows.

The sentence: *same everything as 01 except the tools, and the thing that
makes the new tools possible is a page a person maintains.*

## Pair 1 · Setup (0:00–5:00)

**Diagram — "The same seven pages, plus the index that makes them a wiki" (0:00–2:20)**

- Left: the same seven files. *"Nothing changed here."* Two properties on
  the header: stable names, `##` headings.
- Middle, top: `index.md`. One row per page saying what it is for, plus
  navigation hints. *"This file is written by Knowledge Operations and kept
  true by them. It is not generated. That is the cost of this approach, and
  it is the thing we argued about at 1:10 — same corpus, but the wiki needs
  an index somebody maintains."*
- Middle, bottom: the same question.
- Right: three tools. list the index, grep for a string, read a whole page
  by name. *"The agent reads the index, picks a page by name, reads it
  whole."*
- Bottom strip: the fixed list, unchanged.

**Code (2:20–5:00)** — two cells.

- First cell: identical to 01 plus one line, `WIKI_INDEX`. `Literal` is
  imported for Task 1.
- Second cell: the same model, the same `SYSTEM`. *"No `Document`s, no
  `BM25Retriever`. Nothing is indexed."* Read the print line: same model,
  same prompt.

## Pair 2 · Task 1 (5:00–10:00)

**Diagram — "Three plain functions become three tools, and the schema carries the index" (5:00–7:20)**

- Left: the same `FunctionAgent`. Three arrows to three tools.
- RETURNS column: the index page; `page (line n): text` up to 40 lines; one
  whole page up to 12,000 characters. *"Caps, like the 1,000 in 01. Every
  interface has them."*
- READS column: `index.md`; one page, line by line; `PAGES[page]`. *"Every
  tool starts from a page name. Grep is scoped to a page: you search inside
  a page the caller may read — that's where Nadia's legal-hold rule lives —
  and you never scan the corpus."*
- The line under the box, the important one: `page` is an enum of the
  seven names, **and the descriptions carry the index** — one line per page
  saying what it is for. *"The model knows which page covers tariffs before
  its first call. In 01 it knew nothing. The navigation is in the interface,
  not in a prompt."*
- Bottom: the files, read directly. Nothing indexed.

**Code (7:20–10:00)** — two cells.

- `INDEX_LINE` first: `PAGE_MAP` (the index table as a dict, built in
  Setup) joined into one string, *"Pages: glossary — codes…; tariffs_and_slas
  — pricing, minimum terms…"*. Then the three functions: `list_pages` is one
  line; `grep_wiki(page, pattern)` walks one page, keeps 40 lines;
  `read_page` is the 12,000 cut. `DESCRIPTIONS`: the index string is
  appended to grep and read.
- The JSON under the cell: find `"enum"` and the seven names, then find
  `"description"` and read the tariffs entry out loud. *"That is the wiki
  inside the schema. Somebody wrote that line in index.md; the model reads
  it before it does anything."*
- The grey boxes: `list_pages()` as the agent sees it; `grep_wiki(tariffs,
  '45')` — the lines with a 45 on that page, with line numbers.
  *"Provenance on every hit, for free."*

**Why grep is scoped — the story to tell if asked (30 s).** The first
version of this notebook had a corpus-wide `grep_wiki(pattern)`. Same model,
same prompt, it never touched the index: three greps — `forty-five pounds`
(no match), `45 pounds` (no match), `£45` (one line) — and it answered from
that line. Reworded twice, exactly like RAG. Give a model a word search over
everything and it searches words. Put the index in the schema and scope the
search to a page, and the first move is a page name. That is the whole
argument of this session in one trace.

## Pair 3 · Task 2 (10:00–15:00)

**Diagram — "The loop runs until the model answers or the budget runs out" (10:00–12:20)**

- *"Same picture as 01. Deliberately."* Walk it in 60 seconds: `run_agent`,
  six calls, two exits.
- Spend the rest on THE TRACE panel, which is the real run: two calls,
  both on `page: tariffs_and_slas` — `grep_wiki(…, "45 pounds")`, no match,
  then `grep_wiki(…, "45")`, one line, 529 chars. *"It never searched the
  corpus and never read the index page. The schema told it which page
  covers tariffs, so its first call was already on the right page. 541
  characters of evidence against 3,584 in 01, and every row names its
  source."* (`read_page` was available; the model preferred the cheaper
  scoped search. Fine: the point is the page name, not the verb.)

**Code (12:20–15:00)** — three cells.

- `DCI_AGENT = FunctionAgent(tools=DCI_TOOLS, …)` — *"this is 01's cell with
  one word changed."* `run_agent` is byte-for-byte the same; do not re-read
  it.
- The run: trace table, then answer. Point at the `args` column: every row
  has a page name in it, not a corpus query. Point at the chars: 12, then
  529 — one line of the right page.
- The one-row table. *"Same four numbers as 01. Same number of calls, a
  sixth of the evidence, and a trace that names its source. Not free:
  somebody keeps `index.md` true, and nobody renames a page."*
- Last line on screen before the break: *"What the wiki costs, what it
  buys."* Then hand to Greg.

## If you run over

Cut, in this order: the `grep_wiki(tariffs, '45')` box (say "grep, one
page, with line numbers"); the walk of the loop diagram (say "same as 01"
and go straight to the trace panel); the why-grep-is-scoped story.

Never cut: the index-is-maintained-by-a-person point, the index inside the
JSON description, the trace table.
