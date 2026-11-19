import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium", app_title="02 · LLM Wiki")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 02 · LLM Wiki, and Direct Corpus Interaction

    Same seven pages as notebook 01, same model, same system prompt, same
    loop, same question. This time no retriever picks chunks for the model.
    The agent gets **three tools to move around the pages itself**: list the
    index, grep for a string, read a whole page by name.

    Two terms, used precisely:

    - **LLM Wiki:** Markdown pages with stable names, headings, and an index
      page that says what each page is for. Nadia's pages already have this
      shape, and somebody maintains the index. Without it, this notebook is
      not possible.
    - **Direct Corpus Interaction (DCI):** the agent lists, searches and
      reads those pages directly, deciding what to look at next.

    | | Diagram | Code |
    | --- | --- | --- |
    | **Setup** | the same seven pages, plus the index that makes them a wiki | load the pages and the index, configure the model |
    | **Task 1** | three tools from three plain functions | `list_pages`, `grep_wiki`, `read_page`, and the schema the model sees |
    | **Task 2** | the same loop, a different route | run the same question, read the trace |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 🧑‍💻 Setup

    The corpus is unchanged. What is new on the diagram is `index.md`: one
    row per page saying what it is for, and a short list of navigation
    hints. That file is written and kept current by a person. It is the
    cost of this approach, and it is what the agent reads first.
    """)
    return


@app.cell(hide_code=True)
def _(ui):
    ui.diagram("02_LLM_Wiki_setup")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The same loading cell as notebook 01, with two extra lines: `WIKI_INDEX`
    is the index page, read once, and `PAGE_MAP` is its table as a dict,
    page name → what it is for. Task 1 puts that dict into the tool schema.
    `Literal` is imported because Task 1 uses it to tell the model exactly
    which page names exist.
    """)
    return


@app.cell
def _(mo):
    import json
    import re
    import sys
    import time
    from pathlib import Path
    from typing import Literal

    sys.path.insert(0, str(Path(mo.notebook_dir()).parent))
    from helpers import nb, ui

    CFG = nb.bootstrap(require=("OPENAI_API_KEY",))
    DATA = Path(mo.notebook_dir()) / "data"
    WIKI_INDEX = (DATA / "wiki" / "index.md").read_text(encoding="utf-8")
    PAGE_MAP = dict(re.findall(r"^\| `([^`]+)` \| ([^|]+?) \|$", WIKI_INDEX, re.M))  # page name -> what it is for
    PAGES = {
        page.name: page.read_text(encoding="utf-8")
        for page in sorted((DATA / "corpus").iterdir())
        if page.suffix in {".md", ".txt", ".csv"}
    }
    QUESTION = "Which plan costs forty-five pounds a month, and how long am I tied in for?"

    print(f"✅ model: {CFG}   {len(PAGES)} pages + index.md ({len(PAGE_MAP)} rows)   question: {QUESTION!r}")
    return (
        CFG,
        Literal,
        PAGES,
        PAGE_MAP,
        QUESTION,
        WIKI_INDEX,
        json,
        re,
        time,
        ui,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Identical to notebook 01: the model, the system prompt, the agent class.
    No `Document`s and no `BM25Retriever` this time, because nothing is
    indexed. The tools read the files themselves.
    """)
    return


@app.cell
def _(CFG):
    from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult
    from llama_index.core.tools import FunctionTool
    from llama_index.core.workflow import WorkflowRuntimeError
    from llama_index.llms.litellm import LiteLLM

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
    print(f"✅ same model, same system prompt as 01: {SYSTEM[:60]}…")
    return (
        FunctionAgent,
        FunctionTool,
        LLM,
        SYSTEM,
        ToolCallResult,
        WorkflowRuntimeError,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## ✅ Task 1: Three tools from three plain functions

    **DCI gets tools to move around the pages.** `list_pages` returns the
    index with its navigation hints. `grep_wiki` searches one named page and
    returns matching lines with line numbers. `read_page` returns a complete
    page by its stable name. Every tool starts from a page name, so the
    agent decides where to look before it looks; no retriever decides for it.
    """)
    return


@app.cell(hide_code=True)
def _(ui):
    ui.diagram("02_LLM_Wiki_task1")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The three boxes in the diagram are the three functions below. Nothing is
    embedded, nothing is downloaded, and every tool result is a string a
    person can read.

    The model never sees this Python. What it gets is a **menu**: for each
    tool, its name, a sentence saying what it does, and the values it will
    accept. (The technical name for that menu is the *schema*.) Three things
    about how we wrote the menu do most of the work:

    - **The page name is a fixed list, not free text.** `page` can only be
      one of the seven file names. In Python that rule is written as a
      `Literal`; on the menu it shows up as an `enum`, a closed list of
      allowed values. It cannot invent a page.
    - **The menu says what each page is for.** `read_page`'s description
      includes the index, one line per page: *tariffs — pricing, minimum
      terms*, and so on. So the model knows which page covers tariffs
      **before it makes its first call**.
    - **Search is limited to one page at a time.** `grep_wiki` needs a page
      name before a pattern. That keeps every search inside a page this user
      is allowed to read (which is exactly where a legal-hold rule would sit),
      and it never scans the whole corpus. A search over everything is just
      keyword search again, and a model given one falls back to using it like
      the RAG agent did.

    `grep_wiki` returns at most 40 lines, the way `search_chunks` returned at
    most 1,000 characters.
    """)
    return


@app.cell
def _(FunctionTool, Literal, PAGES, PAGE_MAP, WIKI_INDEX, json, mo, re):
    PageName = Literal[tuple(PAGES)]
    INDEX_LINE = "Pages: " + "; ".join(f"{name} — {use}" for name, use in PAGE_MAP.items())

    def list_pages() -> str:
        return WIKI_INDEX

    def grep_wiki(page: PageName, pattern: str) -> str:
        if page not in PAGES:
            return f"(unknown page: {page})"
        try:
            rx = re.compile(pattern, re.I)
        except re.error:
            rx = re.compile(re.escape(pattern), re.I)
        hits = [f"{page} (line {number}): {line}" for number, line in enumerate(PAGES[page].splitlines(), 1) if rx.search(line)]
        return "\n".join(hits[:40]) or "(no matches)"

    def read_page(page: PageName) -> str:
        if page not in PAGES:
            return f"(unknown page: {page})"
        return PAGES[page][:12000]

    DESCRIPTIONS = {
        "list_pages": "Read the wiki index: what each page is for, and how to navigate between pages.",
        "grep_wiki": "Search one wiki page for exact text or a regular expression. Returns matching lines with line numbers. " + INDEX_LINE,
        "read_page": "Read one complete wiki page by its stable name. " + INDEX_LINE,
    }
    DCI_TOOLS = [FunctionTool.from_defaults(tool, description=DESCRIPTIONS[tool.__name__]) for tool in (list_pages, grep_wiki, read_page)]

    mo.md(f"**What the model receives for `read_page`:**\n\n```json\n{json.dumps(DCI_TOOLS[2].metadata.to_openai_tool(), indent=2)}\n```")
    return DCI_TOOLS, grep_wiki, list_pages


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Two of the tools called by hand, before any model is involved. The index
    as `list_pages` returns it; then a grep scoped to the tariffs page, which
    shows the provenance every hit carries.
    """)
    return


@app.cell
def _(grep_wiki, list_pages, mo):
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

    mo.vstack([
        tool_box("list_pages() — the index, as the agent sees it", list_pages()[:900]),
        tool_box("grep_wiki('northline_tariffs_and_slas.md', '*-') — the lines with a 45 in them, on that page only", grep_wiki("northline_tariffs_and_slas.md", "45")[:900]),
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## ✅ Task 2: The same loop, a different route

    Same `FunctionAgent`, same `run_agent`, same budget of six. Only the
    `tools=` list differs from notebook 01, so the difference in the trace
    is the interface, not the orchestration.
    """)
    return


@app.cell(hide_code=True)
def _(ui):
    ui.diagram("02_LLM_Wiki_task2")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This cell is notebook 01's Task 2 cell with one line changed:
    `tools=DCI_TOOLS`. The trace panel on the right of the diagram is what
    `run_agent` returns: one row per `ToolCallResult`, then the answer, the
    evidence read, the latency and how it stopped.
    """)
    return


@app.cell
def _(
    DCI_TOOLS,
    FunctionAgent,
    LLM,
    SYSTEM,
    ToolCallResult,
    WorkflowRuntimeError,
    time,
):
    DCI_AGENT = FunctionAgent(tools=DCI_TOOLS, llm=LLM, system_prompt=SYSTEM, streaming=False)
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

    print(f"✅ DCI agent: {[tool.metadata.name for tool in DCI_TOOLS]} · budget {MAX_TURNS}")
    return DCI_AGENT, run_agent


@app.cell
def _(CFG, QUESTION, mo):
    run_button = mo.ui.run_button(label=f"▶️  Ask the DCI Wiki agent with {CFG.llm_model}")
    mo.vstack([run_button, mo.md(f"<sub>*{QUESTION}* — up to six model calls.</sub>")])
    return (run_button,)


@app.cell
async def _(DCI_AGENT, QUESTION, mo, run_agent, run_button, ui):
    mo.stop(not run_button.value, mo.callout(mo.md("Press the button to run the question through the agent."), kind="info"))

    with mo.status.spinner(title="Running the agent…"):
        RESULT = await run_agent(QUESTION, DCI_AGENT)

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
    Read the trace before the answer. Every call names a page: the schema
    told the model which page covers tariffs before it made its first call,
    so there was nothing to reword. If it read the whole page, it read more
    characters than the RAG agent did, and the right ones first time; if it
    grepped, it grepped one page with line numbers. **Same question, same
    model, a different route, and the route names its source.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 📋 The row for the comparison

    The same four numbers as notebook 01's row. Put the two together and the
    trade is visible: one side rewords a query until the words match; the
    other names a page and reads it. Neither is free.
    """)
    return


@app.cell
def _(RESULT, ui):
    ui.table(
        [{"interface": "DCI Wiki · list / grep / read", "tool calls": len(RESULT["trace"]),
          "evidence chars": RESULT["evidence_chars"], "latency s": round(RESULT["latency_s"], 1), "stopped": RESULT["stopped"]}],
        title="DCI Wiki, one question",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What the wiki costs:** an index page somebody maintains, stable page
    names nobody renames, and headings worth reading. **What it buys:** a
    trace that names the page it read, and an agent that cannot be sent to a
    page that does not exist.
    """)
    return


if __name__ == "__main__":
    app.run()
