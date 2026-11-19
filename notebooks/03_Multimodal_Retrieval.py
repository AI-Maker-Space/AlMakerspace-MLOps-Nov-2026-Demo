import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium", app_title="03 · Multimodal Retrieval")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 03 · Multimodal Retrieval

    Nadia's tickets are not all text. Attached to them are screenshots,
    scanned runbooks and photographs, and the number that answers a ticket
    is often in the picture, not in any page. This notebook indexes six
    memos and six charts in one vector database, finds the right chart for a
    text question, and reads the number off it.

    The pattern is **separate stores, fused by rank, then read the pixels**:
    each modality gets its own embedder and its own store, the two rankings
    are merged by position rather than score, and a vision model reads only
    the three files that survive.

    | | Diagram | Code |
    | --- | --- | --- |
    | **Setup** | twelve files, and which numbers live only in pixels | load memos as text and charts as images; the two embedders; Qdrant |
    | **Task 1** | two rulers, one ranking | one index, two collections, reciprocal rank fusion |
    | **Task 2** | narrow to three, then read | the vision model answers from memo text and chart pixels |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 🧑‍💻 Setup

    Six charts, six memos, one fictional operator. Four of the six chart
    numbers appear in **no memo**: ticket volume, churn, latency and the
    root-cause split exist only as pixels. A text-only pipeline cannot answer
    a question about them at all.
    """)
    return


@app.cell(hide_code=True)
def _(ui):
    ui.diagram("03_Multimodal_Retrieval_setup")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The top path of the diagram is `SimpleDirectoryReader` over `text/`; the
    bottom path is `ImageDocument` over `images/`. The charts must be
    `ImageDocument`s. Point the reader at the PNGs and it returns plain
    `Document`s that a text embedder will index without ever seeing the
    picture.
    """)
    return


@app.cell
def _(mo):
    import base64
    import json
    import sys
    import warnings
    from pathlib import Path

    import litellm
    from llama_index.core import Document, SimpleDirectoryReader
    from llama_index.core.schema import ImageDocument

    warnings.filterwarnings("ignore", message="Payload indexes have no effect")  # Qdrant in-memory client; harmless

    sys.path.insert(0, str(Path(mo.notebook_dir()).parent))
    from helpers import nb, ui

    CFG = nb.bootstrap(require=("OPENAI_API_KEY", "VLM_MODEL"))
    DATA = Path(mo.notebook_dir()) / "data" / "multimodal"
    VLM = CFG.vlm_kwargs()

    def file_name_only(path: str) -> dict:
        """The only metadata a memo carries. LlamaIndex leaves `file_name` out of the embedding."""
        return {"file_name": Path(path).name}

    memos = SimpleDirectoryReader(DATA / "text", file_metadata=file_name_only).load_data()
    chart_paths = sorted((DATA / "images").glob("*.png"))
    charts = [ImageDocument(image_path=str(path), metadata={"file_name": path.name}) for path in chart_paths]
    DICTIONARY = json.loads((DATA / "data_dictionary.json").read_text(encoding="utf-8"))

    mo.vstack([
        ui.table(
            [{"file": name, "about": info["about"], "the question it answers": info["demo_question"],
              "only in pixels?": "✅" if info["pixels_only"] else "also in a memo"}
             for name, info in DICTIONARY["images"].items()],
            title=f"{len(memos)} memos and {len(charts)} charts · vision model: {CFG.vlm_model}",
        ),
        mo.hstack([mo.image(str(path), width=200) for path in chart_paths], justify="start", wrap=True),
    ])
    return CFG, DATA, Document, ImageDocument, Path, VLM, base64, charts, json, litellm, memos, ui


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Plumbing, not walked on stage.** Two embedding models, both local as
    ONNX through FastEmbed: **bge-small** for memo text, and **CLIP**, whose
    two towers put words and pixels into one 512-dimensional space. The two
    classes tell LlamaIndex how to turn text, or an image path, into a
    vector. Then one in-memory **Qdrant**, a real engine with the same API as
    a production cluster, and two tiny retriever classes that let Task 1
    search the memo half and the chart half of one index separately.
    """)
    return


@app.cell
def _(Path, mo):
    from fastembed import ImageEmbedding, TextEmbedding
    from llama_index.core import StorageContext
    from llama_index.core.embeddings import MultiModalEmbedding
    from llama_index.core.indices import MultiModalVectorStoreIndex
    from llama_index.core.llms import MockLLM
    from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
    from llama_index.core.schema import ImageNode
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

    FASTEMBED_CACHE = str(Path.home() / ".cache" / "fastembed")

    class LocalText(MultiModalEmbedding):
        """bge-small for memos, on this machine. Text only."""

        def model_post_init(self, context):
            object.__setattr__(self, "tower", TextEmbedding("BAAI/bge-small-en-v1.5", cache_dir=FASTEMBED_CACHE))

        def _get_text_embedding(self, text):
            return next(iter(self.tower.embed([text]))).tolist()

        def _get_query_embedding(self, query):
            return self._get_text_embedding(query)

        async def _aget_query_embedding(self, query):
            return self._get_text_embedding(query)

        def _get_image_embedding(self, img_file_path):
            raise NotImplementedError("a text model cannot see")

        async def _aget_image_embedding(self, img_file_path):
            raise NotImplementedError("a text model cannot see")

    class Clip(LocalText):
        """CLIP's two towers: words and pixels into the same 512-dimensional space."""

        def model_post_init(self, context):
            object.__setattr__(self, "tower", TextEmbedding("Qdrant/clip-ViT-B-32-text", cache_dir=FASTEMBED_CACHE))
            object.__setattr__(self, "eye", ImageEmbedding("Qdrant/clip-ViT-B-32-vision", cache_dir=FASTEMBED_CACHE))

        def _get_text_embedding(self, text):
            return next(iter(self.tower.embed([text[:300]]))).tolist()  # CLIP reads 77 tokens

        def _get_image_embedding(self, img_file_path):
            return next(iter(self.eye.embed([str(img_file_path)]))).tolist()

        async def _aget_image_embedding(self, img_file_path):
            return self._get_image_embedding(img_file_path)

    class MemoSearch(BaseRetriever):
        """The text store of a multimodal index, as a retriever of its own."""

        def __init__(self, both):
            super().__init__()
            self.both = both

        def _retrieve(self, query_bundle):
            return self.both.text_retrieve(query_bundle)

    class ChartSearch(MemoSearch):
        """The image store, searched with the question through CLIP's text tower."""

        def _retrieve(self, query_bundle):
            return self.both.text_to_image_retrieve(query_bundle)

    with mo.status.spinner(title="Loading the local embedding models (downloads once, ~650 MB)…"):
        TEXT_EMBED = LocalText()
        CLIP = Clip()
    QDRANT = QdrantClient(":memory:")
    print("✅ text: bge-small-en-v1.5 (384 d)   CLIP: ViT-B/32 (512 d)   Qdrant: in-memory")
    return (
        CLIP,
        ChartSearch,
        ImageNode,
        MemoSearch,
        MockLLM,
        MultiModalVectorStoreIndex,
        QDRANT,
        QdrantVectorStore,
        QueryFusionRetriever,
        StorageContext,
        TEXT_EMBED,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## ✅ Task 1: Two rulers, one ranking

    Each modality gets its best embedder: bge-small for the memos, CLIP for
    the charts, in two Qdrant collections behind one index. The two stores'
    scores live on different scales, so the lists cannot be merged by score.
    They are merged by **rank**: Reciprocal Rank Fusion gives an item
    `1 / (60 + rank)` from every list it appears in and sorts by the sum.
    """)
    return


@app.cell(hide_code=True)
def _(ui):
    ui.diagram("03_Multimodal_Retrieval_task1")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Think of two filing cabinets: one for memos, one for charts.

    - `MultiModalVectorStoreIndex` owns both cabinets at once. `vector_store=`
      is the memo cabinet, `image_store=` is the chart cabinet.
    - `as_retriever` is the clerk. Give it a question and it pulls the three
      best memos from one cabinet and the three best charts from the other.
    - `MemoSearch` and `ChartSearch` are two small helpers that let us ask
      each cabinet on its own, so the two lists can be seen side by side.
    - `QueryFusionRetriever` merges the two lists into one.
      `mode="reciprocal_rerank"` means: ignore the scores (they are on
      different scales) and compare *positions* — first place on either list
      counts the same. It would normally ask a language model to rewrite the
      question a few ways; `num_queries=1` switches that off, so the
      `MockLLM` it still asks for is a stand-in that is never used.

    The first table is each cabinet's own scores. The second is the one
    merged list, shown as a share of the top score so the numbers are
    readable.
    """)
    return


@app.cell
def _(
    CLIP,
    ChartSearch,
    MemoSearch,
    MockLLM,
    MultiModalVectorStoreIndex,
    QDRANT,
    QdrantVectorStore,
    QueryFusionRetriever,
    StorageContext,
    TEXT_EMBED,
    charts,
    memos,
    mo,
    ui,
):
    index = MultiModalVectorStoreIndex.from_documents(
        memos + charts,
        storage_context=StorageContext.from_defaults(
            vector_store=QdrantVectorStore(client=QDRANT, collection_name="memos"),
            image_store=QdrantVectorStore(client=QDRANT, collection_name="charts"),
        ),
        embed_model=TEXT_EMBED, image_embed_model=CLIP,
    )
    both = index.as_retriever(similarity_top_k=3, image_similarity_top_k=3)
    retriever = QueryFusionRetriever(
        [MemoSearch(both), ChartSearch(both)], mode="reciprocal_rerank",
        similarity_top_k=3, num_queries=1, llm=MockLLM(), use_async=False,
    )

    def rows(hits):
        return [{"rank": rank, "file": hit.node.metadata["file_name"], "kind": "chart" if hasattr(hit.node, "image_path") else "memo",
                 "score": round(hit.score, 3)} for rank, hit in enumerate(hits, start=1)]

    def fused_rows(hits):
        """Rank-fusion scores are tiny fractions (1/61 for first place on one list). Show them as a share of the top score."""
        top = hits[0].score if hits else 1
        return [{"rank": rank, "file": hit.node.metadata["file_name"], "kind": "chart" if hasattr(hit.node, "image_path") else "memo",
                 "fused score (% of top)": f"{100 * hit.score / top:.0f}%"} for rank, hit in enumerate(hits, start=1)]

    question_1 = "why did the on-call engineer get paged about the Coast region?"
    mo.vstack([
        ui.table(rows(both.retrieve(question_1)), title=f"Raw scores · '{question_1}' · two embedders, two scales"),
        ui.table(fused_rows(retriever.retrieve(question_1)), title="Fused by rank · the best memo and the best chart both make the top 3"),
    ])
    return (retriever,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Look at the raw scores first. Memos score around 0.6 under bge-small and
    charts around 0.3 under CLIP, but those are two different rulers. Sorted
    by them, every memo beats every chart, every time. The fused ranking
    ignores the numbers and keeps only the order.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## ✅ Task 2: Narrow to three, then read

    CLIP can find the chart; it cannot read the number off it. The vision
    model's job is the second half: the prompt carries the retrieved memos as
    text and the retrieved charts as **the actual pixels**, loaded from disk,
    and the model is asked to cite the files it used.
    """)
    return


@app.cell(hide_code=True)
def _(ui):
    ui.diagram("03_Multimodal_Retrieval_task2")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `answer` is the whole right half of the diagram. `retriever.retrieve`
    narrows twelve files to three; the loop builds one message with a text
    part per memo and an `image_url` part per chart; one `litellm.completion`
    reads it. The shape matters more than the prompt: **the vision model
    reads three candidates, never the corpus.** At nine hundred screenshots,
    a query costs three image reads instead of nine hundred.
    """)
    return


@app.cell
def _(ImageNode, VLM, base64, litellm, retriever):
    ANSWER_SYSTEM = (
        "Answer ONLY from the provided context (text and images). Read any figures carefully "
        "for exact values. Cite the files you used like [file]. If the answer isn't there, say so."
    )

    def image_part(path: str) -> dict:
        """One image, as the content part every OpenAI-compatible endpoint accepts."""
        encoded = base64.b64encode(open(path, "rb").read()).decode()
        mime = "jpeg" if str(path).lower().endswith((".jpg", ".jpeg")) else "png"
        return {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{encoded}"}}

    def answer(question: str, narrow=retriever) -> tuple[str, list]:
        """Narrow with the fused retriever, then hand the model the text and the pixels of what survived."""
        hits = narrow.retrieve(question)
        parts = [{"type": "text", "text": f"Question: {question}\n\n--- CONTEXT ---"}]
        for hit in hits:
            name = hit.node.metadata["file_name"]
            if isinstance(hit.node, ImageNode):
                parts += [{"type": "text", "text": f"\n[{name}] (chart below):"}, image_part(hit.node.image_path)]
            else:
                parts.append({"type": "text", "text": f"\n[{name}]: {hit.node.get_content()}"})
        reply = litellm.completion(
            messages=[{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": parts}],
            timeout=600, **VLM,
        )
        return reply.choices[0].message.content or "", hits

    return (answer,)


@app.cell
def _(CFG, mo):
    answer_button = mo.ui.run_button(label=f"▶️  Answer two questions with {CFG.vlm_model}")
    mo.vstack([answer_button, mo.md("<sub>Two model calls, each carrying up to three images. The first answer lives only in a chart.</sub>")])
    return (answer_button,)


@app.cell
def _(answer, answer_button, mo):
    mo.stop(not answer_button.value, mo.callout(mo.md("Press the button to retrieve, then read."), kind="info"))

    DEMO_QUESTIONS = [
        "Which quarter had the most support tickets, and how many?",  # only in a chart
        "In which month was churn highest, and what caused it?",  # a chart and a memo together
    ]
    with mo.status.spinner(title="Retrieving three files per question, then reading…"):
        answered = [(question, *answer(question)) for question in DEMO_QUESTIONS]

    mo.md("\n\n---\n\n".join(
        f"**Q: {question}**\n\n{text}\n\n<sub>read: {', '.join(hit.node.metadata['file_name'] for hit in hits)}</sub>"
        for question, text, hits in answered
    ))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice the division of labour: **retrieval** finds the tickets chart, and
    **reading** gets the number off the bar. That number is in no memo.
    *Retrieve to find the chart, look at the chart to answer* is the pattern,
    and `k=3` is a cost control as much as a quality control: every image in
    the prompt is roughly a thousand tokens.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 🎁 Bonus · Task 3: Video is the same two stores, keyed by time

    *Not part of the 15 minutes. Cut it first if time is short.*

    A support call recording, or this 45-second narrated review, is text and
    pictures in a time series. Speech-to-text gives transcript segments with
    timestamps; one frame per segment gives the pictures. Nothing else
    changes: segments go in the text store, frames in the CLIP store, both
    named by their timestamp, and the fusion and the read step are the ones
    from Tasks 1 and 2. The citation that comes back is a timestamp.
    """)
    return


@app.cell(hide_code=True)
def _(ui):
    ui.diagram("03_Multimodal_Retrieval_task3")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `stamp` is the whole trick: the same `file_name` on a segment's text and
    on its frame, `review.mp4 @ 26–37 s`, so whichever store finds it, the
    citation names a moment in the video. The index, the retriever and
    `answer` are the Task 1 and Task 2 code with a different list of records.
    """)
    return


@app.cell
def _(
    CLIP,
    ChartSearch,
    DATA,
    Document,
    ImageDocument,
    MemoSearch,
    MockLLM,
    MultiModalVectorStoreIndex,
    QDRANT,
    QdrantVectorStore,
    QueryFusionRetriever,
    StorageContext,
    TEXT_EMBED,
    answer,
    json,
    mo,
    ui,
):
    VIDEO = DATA / "video"
    TRANSCRIPT = json.loads((VIDEO / "review_transcript.json").read_text(encoding="utf-8"))["segments"]

    def stamp(seg: dict) -> str:
        return f"review.mp4 @ {seg['start']:.0f}–{seg['end']:.0f} s"

    def segment_at(second: int) -> dict:
        return next(seg for seg in TRANSCRIPT if seg["start"] <= second < seg["end"])

    segments = [Document(text=seg["text"], metadata={"file_name": stamp(seg)}, excluded_embed_metadata_keys=["file_name"])
                for seg in TRANSCRIPT]
    frames = [ImageDocument(image_path=str(path), metadata={"file_name": stamp(segment_at(int(path.stem.split("_")[1])))})
              for path in sorted((VIDEO / "keyframes").glob("*.jpg"))]  # frame_0031.jpg is the frame at 31 s

    index_v = MultiModalVectorStoreIndex.from_documents(
        segments + frames,
        storage_context=StorageContext.from_defaults(
            vector_store=QdrantVectorStore(client=QDRANT, collection_name="video_transcript"),
            image_store=QdrantVectorStore(client=QDRANT, collection_name="video_frames"),
        ),
        embed_model=TEXT_EMBED, image_embed_model=CLIP,
    )
    both_v = index_v.as_retriever(similarity_top_k=3, image_similarity_top_k=3)
    retriever_v = QueryFusionRetriever(
        [MemoSearch(both_v), ChartSearch(both_v)], mode="reciprocal_rerank",
        similarity_top_k=3, num_queries=1, llm=MockLLM(), use_async=False,
    )

    video_question = "Why did the Coast region page the on-call engineer, and where in the review is that covered?"
    video_text, video_hits = answer(video_question, retriever_v)
    mo.vstack([
        ui.table([{"start": f"{seg['start']:.0f} s", "end": f"{seg['end']:.0f} s", "record name": stamp(seg), "transcript": seg["text"][:70] + "…"}
                  for seg in TRANSCRIPT], title=f"{len(segments)} transcript segments and {len(frames)} keyframes, named by timestamp"),
        mo.md(f"**Q: {video_question}**\n\n{video_text}\n\n<sub>read: {', '.join(hit.node.metadata['file_name'] for hit in video_hits)}</sub>"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `read:` line is the point. Every record the model saw is a moment in
    the video, so the citation is something a person can scrub to. For
    Nadia's support calls: a transcript per call, chunked by time window, a
    representative frame per window, and a timestamp on every citation. Same
    two stores, same fusion, same three reads per query.
    """)
    return


if __name__ == "__main__":
    app.run()
