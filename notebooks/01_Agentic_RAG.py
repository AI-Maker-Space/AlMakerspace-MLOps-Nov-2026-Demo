import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium", app_title="01 · Agentic RAG")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 01 · Agentic RAG

    Nadia is a second-line support engineer at a regional broadband operator.
    A ticket comes in, and the answer is never in the ticket: it is in a
    runbook, a tariff page, or the glossary. Her knowledge base is seven
    Markdown pages. This notebook gives an agent **one tool** over them: a
    ranked chunk search it can call as many times as it likes.

    Three steps, each one diagram followed by the code it draws:

    | | Diagram | Code |
    | --- | --- | --- |
    | **Setup** | the seven pages, and the question we will ask | load the pages, configure the model |
    | **Task 1** | pages → sections → BM25 index → `search_chunks` | build the index and the tool |
    | **Task 2** | the agent loop, and the trace it leaves | run one question, read the trace |

    Notebook 02 asks the **same question** of the **same pages** with different
    tools. Everything else, the model, the system prompt and the loop, is
    identical, so the difference you see there comes from the interface alone.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 🧑‍💻 Setup

    Seven files, one question. The question is chosen because it shares no
    words with the page that answers it: there is no `£45`, no `tariff`, no
    `minimum term` in it. A ranked search has nothing to rank on until the
    agent rewords it.
    """)
    return


@app.cell(hide_code=True)
def _(ui):
    ui.diagram("01_Agentic_RAG_setup")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The left column of the diagram is this cell: each file becomes one
    `PAGES` entry, keyed by its stable name. `CFG` reads `.env`, so the model
    is configuration, not code.
    """)
    return


@app.cell
def _(mo):
    import json
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(mo.notebook_dir()).parent))
    from helpers import nb, ui

    CFG = nb.bootstrap(require=("OPENAI_API_KEY",))
    DATA = Path(mo.notebook_dir()) / "data"
    PAGES = {
        page.name: page.read_text(encoding="utf-8")
        for page in sorted((DATA / "corpus").iterdir())
        if page.suffix in {".md", ".txt", ".csv"}
    }
    QUESTION = "Which plan costs forty-five pounds a month, and how long am I tied in for?"

    print(f"✅ model: {CFG}   {len(PAGES)} pages   question: {QUESTION!r}")
    return CFG, PAGES, QUESTION, json, time, ui


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The right column of the diagram is this cell: the model behind LiteLLM,
    the system prompt the agent runs under, and the pages as LlamaIndex
    `Document`s. `excluded_embed_metadata_keys` keeps the page name for
    citing but out of the index, so every section of the tariffs page does
    not match the word *tariffs*.
    """)
    return


@app.cell
def _(CFG, PAGES):
    from llama_index.core import Document
    from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult
    from llama_index.core.node_parser import MarkdownNodeParser
    from llama_index.core.tools import FunctionTool
    from llama_index.core.workflow import WorkflowRuntimeError
    from llama_index.llms.litellm import LiteLLM
    from llama_index.retrievers.bm25 import BM25Retriever

    class ToolCallingLiteLLM(LiteLLM):
        """LiteLLM, told that this endpoint can call tools."""

        @property
        def metadata(self):
            return super().metadata.model_copy(update={"is_function_calling_model": True})

    llm_kwargs = CFG.kwargs()
    LLM = ToolCallingLiteLLM(
        model=llm_kwargs.pop("model"), temperature=0, max_retries=2,
        additional_kwargs={**llm_kwargs, "timeout": 600},
    )
    SYSTEM = (
        "Answer Northline Broadband questions only from tool evidence. Investigate before answering. "
        "Cite page names. If evidence is incomplete, say so."
    )
    DOCS = [
        Document(text=text, metadata={"page": name}, excluded_embed_metadata_keys=["page", "header_path"])
        for name, text in PAGES.items()
    ]
    print(f"✅ {len(DOCS)} Documents · system prompt: {SYSTEM[:60]}…")
    return (
        BM25Retriever,
        DOCS,
        FunctionAgent,
        FunctionTool,
        LLM,
        MarkdownNodeParser,
        SYSTEM,
        ToolCallResult,
        WorkflowRuntimeError,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## ✅ Task 1: The pages as sections, and a BM25 retriever

    **Agentic RAG receives ranked chunks.** Each page is split at its
    headings, BM25 indexes the sections, and the one tool the agent has
    returns the top-k sections for a query. It can search again. It cannot
    list files or ask for a whole page.
    """)
    return


@app.cell(hide_code=True)
def _(ui):
    ui.diagram("01_Agentic_RAG_task1")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The four boxes down the left of the diagram are the four lines of this
    cell: `MarkdownNodeParser` ends a section at the next heading;
    `BM25Retriever` scores sections by the rare words they share with the
    query; `search_chunks` returns at most eight sections, each cut at 1,000
    characters. That cut is the dashed line on the right of the diagram.
    **Every chunk retriever has a cap somewhere.** Here it is visible.
    """)
    return


@app.cell
def _(BM25Retriever, DOCS, MarkdownNodeParser, PAGES, mo, ui):
    nodes = MarkdownNodeParser().get_nodes_from_documents(DOCS)
    bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=8)

    def search_chunks(query: str, k: int = 4) -> str:
        """Search ranked corpus chunks. Search again with a new query when evidence is incomplete."""
        hits = bm25.retrieve(query)[: max(1, min(k, 8))]
        return "\n\n---\n\n".join(f"[{hit.node.metadata['page']}]\n{hit.node.text[:1000]}" for hit in hits)

    mo.vstack([
        ui.table(
            [{"page": name, "sections": sum(1 for node in nodes if node.metadata["page"] == name),
              "characters": len(text), "longest section": max(len(node.text) for node in nodes if node.metadata["page"] == name)}
             for name, text in PAGES.items()],
            title=f"The index: {len(nodes)} sections from {len(PAGES)} pages",
        ),
        ui.table(
            [{"rank": rank, "page": hit.node.metadata["page"], "section": hit.node.text.splitlines()[0], "score": round(hit.score, 2)}
             for rank, hit in enumerate(bm25.retrieve("LOS 0x21")[:3], start=1)],
            title="bm25.retrieve('LOS 0x21'), top 3 — a query that shares words with its answer",
        ),
    ])
    return (search_chunks,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What the agent actually receives is one string. This is `search_chunks`
    called by hand with our question, before any model is involved. Read the
    section headings: the question's words reach the tariffs page, but the
    wrong sections of it. Nothing here says *Fibre 500* or *24 months*.
    """)
    return


@app.cell
def _(QUESTION, mo, search_chunks):
    def tool_box(title: str, text: str):
        """A tool result as the agent receives it: a string, wrapped so it stays readable."""
        escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return mo.vstack([
            mo.md(f"**{title}**"),
            mo.Html(
                f'<pre style="white-space:pre-wrap; word-break:break-word; font-size:0.85em; '
                f'padding:0.75em; border-radius:6px; background:rgba(127,127,127,0.08)">{escaped}</pre>'
            ),
        ])

    tool_box(f"search_chunks({QUESTION!r}, k=3) — what the agent sees", search_chunks(QUESTION, k=3)[:1200])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## ✅ Task 2: One agent, one tool, and its trace

    `FunctionAgent` is the loop. It sends the question and the tool schema to
    the model, runs whatever tool the model asks for, and sends the result
    back. It stops when the model answers without asking for a tool, or when
    the budget runs out.
    """)
    return


@app.cell(hide_code=True)
def _(ui):
    ui.diagram("01_Agentic_RAG_task2")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `FunctionTool.from_defaults` is the box labelled *tool* in the diagram:
    it reads the function's name, type hints and docstring and writes the
    JSON schema the model receives. `run_agent` is the grey frame around the
    loop: it streams the events, keeps every `ToolCallResult` as a row of the
    trace, and records how the run stopped. `max_iterations=6` is the budget.
    LlamaIndex counts every model call against it, the answer included.
    """)
    return


@app.cell
def _(
    FunctionAgent,
    FunctionTool,
    LLM,
    SYSTEM,
    ToolCallResult,
    WorkflowRuntimeError,
    json,
    mo,
    search_chunks,
    time,
):
    RAG_TOOLS = [FunctionTool.from_defaults(search_chunks, description=search_chunks.__doc__)]
    RAG_AGENT = FunctionAgent(tools=RAG_TOOLS, llm=LLM, system_prompt=SYSTEM, streaming=False)
    MAX_TURNS = 6

    async def run_agent(question: str, agent) -> dict:
        """One question through one agent: the answer, the trace, the evidence read, the time, how it stopped."""
        started = time.perf_counter()
        handler = agent.run(question, max_iterations=MAX_TURNS)
        trace = []
        try:
            async for event in handler.stream_events():
                if isinstance(event, ToolCallResult):
                    returned = str(event.tool_output.content)
                    trace.append({"tool": event.tool_name, "args": event.tool_kwargs, "chars": len(returned)})
            answer, stopped = str(await handler).strip(), "answer"
        except WorkflowRuntimeError:
            answer, stopped = "(max turns reached)", "max_turns"
        return {"answer": answer, "trace": trace, "evidence_chars": sum(step["chars"] for step in trace),
                "latency_s": time.perf_counter() - started, "stopped": stopped}

    mo.md(f"**What the model receives for `search_chunks`:**\n\n```json\n{json.dumps(RAG_TOOLS[0].metadata.to_openai_tool(), indent=2)}\n```")
    return RAG_AGENT, run_agent


@app.cell
def _(CFG, QUESTION, mo):
    run_button = mo.ui.run_button(label=f"▶️  Ask the Agentic RAG agent with {CFG.llm_model}")
    mo.vstack([run_button, mo.md(f"<sub>*{QUESTION}* — up to six model calls.</sub>")])
    return (run_button,)


@app.cell
async def _(QUESTION, RAG_AGENT, mo, run_agent, run_button, ui):
    mo.stop(not run_button.value, mo.callout(mo.md("Press the button to run the question through the agent."), kind="info"))

    with mo.status.spinner(title="Running the agent…"):
        RESULT = await run_agent(QUESTION, RAG_AGENT)

    mo.vstack([
        ui.table(
            [{"call": n, "tool": step["tool"], "args": ", ".join(f"{k}={v!r}" for k, v in step["args"].items()), "chars returned": step["chars"]}
             for n, step in enumerate(RESULT["trace"], start=1)],
            title=f"The trace · {len(RESULT['trace'])} tool calls · {RESULT['evidence_chars']:,} evidence chars · "
                  f"{RESULT['latency_s']:.1f} s · stopped: {RESULT['stopped']}",
        ),
        mo.md(f"**Answer:** {RESULT['answer']}"),
    ])
    return (RESULT,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read the trace before the answer. The first query is the model's rewording
    of the question; if the ranked sections did not carry the tariff, there is
    a second query with different words. The agent never sees a page name
    until a chunk happens to carry one. **The trace is what an auditor
    reconstructs, what a reviewer replays, and what a cost line is built
    from.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 📋 The row for the comparison

    One row, four numbers, the same four that notebook 02 produces for the
    same question. The slides put the two rows side by side.
    """)
    return


@app.cell
def _(RESULT, ui):
    ui.table(
        [{"interface": "Agentic RAG · search_chunks", "tool calls": len(RESULT["trace"]),
          "evidence chars": RESULT["evidence_chars"], "latency s": round(RESULT["latency_s"], 1), "stopped": RESULT["stopped"]}],
        title="Agentic RAG, one question",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What stays fixed in 02:** the model, the system prompt, `FunctionAgent`,
    `run_agent`, the budget, the seven files and the question. **What
    changes:** the `tools=` list.
    """)
    return


if __name__ == "__main__":
    app.run()
