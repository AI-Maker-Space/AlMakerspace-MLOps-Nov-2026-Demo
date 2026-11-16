# 03 · Multimodal Retrieval — the 15-minute script

Slot: 2:10–2:25, after the break and Greg's "we've done text and PDF, now
pictures" bridge. Pre-run (press ▶️ before the break; the two vision calls
take ~10 s). Three pairs; ~2:20 per diagram, ~2:40 per code cell. **Hard stop
at 15** — the final discussion follows.

The sentence: *retrieval finds the chart; the vision model reads it; the
model only ever reads three files.*

Strategies A and B are **not** in this notebook. They are on Greg's slides
and in the closing table. If asked, the one-line answers: A captions every
image with a vision model at ingestion (expensive, as good as the caption);
B puts words and pixels in one CLIP space (cheap, cannot read values).

## Pair 1 · Setup (0:00–5:00)

**Diagram — "Four of the six chart numbers exist only as pixels" (0:00–2:20)**

- Left: six memos, six charts. The purple chips: four numbers — churn,
  latency, root cause, tickets — appear in **no memo**. *"A text pipeline
  cannot answer those questions. Not badly — at all."*
- Top path: `SimpleDirectoryReader` over `text/` → six `Document`s.
- Bottom path: `ImageDocument` over `images/` → six image documents.
- The orange ✗: point the reader at PNGs and you get plain `Document`s. *"A
  text embedder indexes those without complaint and without ever seeing the
  picture. The commonest multimodal bug is this one."*

**Code (2:20–5:00)** — two cells, the second is plumbing.

- First cell: `memos` is the top path, `charts` the bottom path. The table
  is the answer key: which question each chart answers and whether the
  number is pixels-only. The six thumbnails.
- Second cell, say it in 40 seconds and do not scroll into it: *"Two local
  embedders through FastEmbed — bge-small for text, CLIP for pixels, no
  torch. One in-memory Qdrant, same API as the cluster. Two tiny retriever
  classes so Task 1 can search the memo half and the chart half separately."*

## Pair 2 · Task 1 (5:00–10:00)

**Diagram — "Two rulers, one ranking: fuse by rank, never by score" (5:00–7:20)**

- The question goes to two stores at once. `MemoSearch` → bge-small → memos
  at ~0.6. `ChartSearch` → CLIP's text tower → charts at ~0.3.
- Orange box: merged by raw score, three memos, no chart. *"Two different
  rulers. Sort by them and every memo beats every chart, every time."*
- Purple box: `reciprocal_rerank`. `1 / (60 + rank)` from every list an
  item appears in, summed. Top of its list is 1/60 whether it scored 0.6 or
  0.3. *"The best memo and the best chart both make the top 3. That is the
  whole trick."*

**Code (7:20–10:00)** — one cell, two tables.

- `MultiModalVectorStoreIndex` with `vector_store=` and `image_store=`: the
  two stores, two Qdrant collections. `embed_model=TEXT_EMBED,
  image_embed_model=CLIP`: one embedder each.
- `as_retriever(similarity_top_k=3, image_similarity_top_k=3)`: top 3 from
  each. `QueryFusionRetriever([...], mode="reciprocal_rerank")`: the fusion.
  `num_queries=1` turns off query rewriting, hence `MockLLM`.
- Table 1, raw scores: read the `score` column — memos ~0.6, charts ~0.3,
  kind column. Table 2, fused: a chart is in the top 3. *"Nothing about the
  numbers changed. Only how they were combined."*

## Pair 3 · Task 2 (10:00–15:00)

**Diagram — "Retrieval narrows twelve files to three; the vision model reads only those" (10:00–12:20)**

- Left: twelve files, three highlighted. The retriever from Task 1, top 3.
- Middle, ONE MESSAGE: the system prompt (context only, cite like
  `[file]`), the question, then each hit in order — a memo as its text, a
  chart as its file name **followed by its pixels** as an `image_url` part.
- Right: the vision model reads that one message. Answer with citations.
- Bottom line: *"Twelve in the corpus, three in the prompt. At nine hundred
  screenshots: three image reads per query, not nine hundred."*

**Code (12:20–15:00)** — two cells.

- `image_part`: base64 of the PNG as an `image_url` part. `answer`: retrieve
  → build `parts` (text for memos, `image_part` for charts) → one
  `litellm.completion` with `**VLM`. *"The shape is the lesson, not the
  prompt."*
- The run, two questions. Question 1 — tickets by quarter — lives only in a
  chart; read the answer, then the `read:` line underneath: the tickets
  chart is in it. *"Retrieval found it. The vision model read 27 thousand
  off the bar. That number is in no memo."* Question 2 — churn — needs a
  chart **and** a memo; the answer cites both.
- Close on the strategy table (do not read it; one sentence): *"We built C.
  The default is C for breadth, with a vision model at answer time so exact
  figures come from pixels. Scans: same three strategies, OCR as the parser.
  Video: transcript chunks by time window in the text store, a frame per
  window in the image store, a timestamp on every citation."* Hand to Greg
  for the which-one-should-I-use discussion.

## Bonus · Task 3: video (not in the 15)

At the bottom of the notebook, after the strategy table. Pre-run with the
rest (it is one more vision call). Cut it first; the closing sentence
already says what it shows. If the room is quick, it is two minutes:

- **Diagram (60 s).** Five timestamped segments across the top. One
  segment becomes two records with the same name, `review.mp4 @ 26–37 s`:
  the transcript text into the text store, the keyframe at 31 s into the
  CLIP store. Right side: the exact pipeline from Tasks 1 and 2. *"Nothing
  new. Two stores, keyed by time."*
- **Code (60 s).** `stamp` is the whole trick — the same `file_name` on the
  text and the frame. `segments`, `frames`, then `index_v` and
  `retriever_v` are Task 1's lines with a different list, and
  `answer(video_question, retriever_v)` is Task 2's function. Read the
  `read:` line under the answer: every record is a moment in the video.
  *"For Nadia's calls: a transcript per call, chunked by time window, a
  frame per window, a timestamp on every citation."*

If the team decides to bring it inside the 15: run 3 / 5 / 4 / 3 — Setup
to three minutes (one sentence on the plumbing cell), Task 2 to one demo
question, video at 12–15 — with the rule *"not at Task 3 by 12:00, skip
it."*

## If you run over

Cut, in this order: the second Setup cell entirely (say "two embedders, one
Qdrant"); the raw-scores table (say "0.6 and 0.3, different rulers" from the
diagram); the second demo question.

Never cut: the pixels-only chips, the orange "merged by raw score" box, the
`read:` line under the first answer.
