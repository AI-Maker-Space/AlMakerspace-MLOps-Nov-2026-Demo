# 01 vs 02 — what actually differs

Background for the comparison slide after notebook 02, and for the question
you will get in the hallway: *"isn't the LLM Wiki just RAG with an index
page?"* No. The index page is the precondition, not the mechanism. Three
things change between 01 and 02, and they stack.

(One correction worth making if someone says it: 01 is not a vector store.
It is a BM25 keyword index. A vector store would be the same shape with
embeddings instead of word overlap, and the agent's situation would be
identical: it only ever sees ranked chunks.)

## 1. Who decides what the model reads

**01 — a retriever decides.** The model types a query, BM25 scores 67
sections against it, and the model gets back the top few, cut at 1,000
characters each. It never chooses a page, never sees a page whole, and never
knows what it *didn't* get. Everything it learns about the corpus arrives
pre-selected and pre-trimmed.

**02 — nothing selects for it.** `list_pages` says what exists, `grep_wiki`
searches one named page, `read_page` hands over the entire page. Every tool
starts from a page name, so the model does the navigation: decide "tariffs",
then read it top to bottom. That is the "direct" in Direct Corpus
Interaction.

## 2. What the tools know about the corpus

**01** — one tool, free-text in, knows nothing about structure. The schema is
`query: string, k: integer`.

**02** — the tools carry the corpus's shape in their definitions. `page` is
a `Literal` of the seven filenames, which becomes an `enum` in the JSON
schema, and the descriptions of `grep_wiki` and `read_page` carry the index
itself: one line per page saying what it is for. The model is told the page
names *and their purposes* before it makes a single call, and cannot ask for
a page that does not exist. The navigation happens in the tool definition,
before the index page is ever read.

This was learned the hard way. The first version had a corpus-wide
`grep_wiki(pattern)` and a `list_pages` the model had to choose to call. It
never chose to: three greps (`forty-five pounds`, `45 pounds`, `£45`), two
misses, one hit, reworded exactly like RAG. Give a model a word search over
everything and it searches words. Structure that lives in instructions is
optional; structure that lives in the interface is not.

## 3. What the trace can tell you afterwards

This is the one Nadia's manager cares about.

**01's trace** is a list of queries the model invented — in the run we made,
`plan costs forty-five pounds a month`, then `plan costs 45 pounds a month` —
and how many characters came back. If the answer is wrong, you re-run the
retriever to find out what the model saw.

**02's trace** names its source on every row — in the run we made,
`grep_wiki(page=northline_tariffs_and_slas.md, "45 pounds")` then the same
page with `"45"`: reproducible, auditable without re-running anything. The
same two calls as 01, a sixth of the evidence (541 characters against
3,584), and a citation.

## Where the index page fits

The index is what makes 2 and 3 *possible*. Without stable names there is no
enum; without an index the model's first move is blind grepping. So it is the
precondition — and the maintenance cost: a person keeps `index.md` true, and
nobody renames a page.

## The sentence for the slide

01 asks *"what text looks like my question?"* 02 asks *"which page would a
person open?"* and opens it. The first works on any pile of documents and
degrades when the question's words don't match. The second needs the pile to
have been organised by someone, and in return the agent behaves like a
colleague who knows where things are.

## What is held fixed, on purpose

The model, the system prompt, `FunctionAgent`, `run_agent`, the six-call
budget, the seven files, the question. When the two rows sit side by side,
the only variable is the interface.

| | 01 Agentic RAG | 02 LLM Wiki / DCI |
| --- | --- | --- |
| Who picks the text | BM25, by shared words | the model, by page name |
| What the model sees | ranked sections, ≤1,000 chars each | whole pages, the index, grep lines with provenance |
| What the tools know | nothing about structure | the seven page names as an enum, and what each is for, in the description |
| What the trace shows | the queries it tried | the page it chose, on every row |
| What it needs from the corpus | nothing | stable names, headings, a maintained index |
| Fails when | the question's words miss the page's words | the index is stale, a page is renamed, or the search is left corpus-wide |
