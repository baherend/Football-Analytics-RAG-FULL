"""
chat.py — Phase 6: Interactive RAG Chat

Terminal-based interface for testing the RAG pipeline interactively.

Usage:
    python chat.py                    # Interactive mode
    python chat.py "your question"    # Single question mode

Commands:
    /context  - Show retrieved context for last question
    /prompt   - Show full prompt sent to LLM
    /route    - Show routing decision for last question
    /mode     - Switch between hybrid/semantic/structured
    /model    - Switch LLM model
    /help     - Show this help
    /quit     - Exit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.artifacts import ArtifactPaths, resolve_runtime_artifact_paths
from src.conversation_memory import (
    ConversationMemory,
    format_conversation_context,
    resolve_pronoun_references,
)
from src.dataset_catalog import discover_datasets


# ---------------------------------------------------------------------------
# Pipeline Integration
# ---------------------------------------------------------------------------


# Load modules
print("Loading RAG pipeline...")
try:
    # Structural Cleanup Phase B: query routing/execution is a real package
    # module now (src/query/router.py, see Phase A) -- loaded with a normal
    # import instead of the file-path loader the numbered-script era needed.
    # `router_mod` is kept as the existing name/attribute-access pattern
    # (router_mod.route_query(...), router_mod.execute_route(...)) so
    # nothing downstream in this file changes.
    import src.query.router as router_mod
    # Phase B1: the answer policy this CLI shares with
    # 07_prompting.py::answer_question() (context assembly, refusal gate,
    # verification + citations). Routing, prompt building and generation stay
    # here -- tests stub them on this module by design.
    import src.orchestration.policy as orchestration
except Exception as e:
    print(f"Error loading router (src.query.router): {e}")
    print("Make sure all dependencies are installed: pip install -r requirements.txt")
    raise SystemExit(1)

try:
    # Phase B0 -- ONE canonical module identity. This used to load
    # 07_prompting.py by file path under the name "prompting", which created a
    # SECOND module instance: `chat.prompting_mod is import_module("07_prompting")`
    # measured False, so module-level state and monkeypatches applied to only
    # one copy. `07_prompting` is the canonical name (28 call sites across
    # streamlit_app.py and tests use it; only this file used "prompting"), and
    # import_module() accepts it despite the leading digit. The file-path
    # loader that created the duplicate was removed with this change.
    from importlib import import_module as _import_module

    prompting_mod = _import_module("07_prompting")
except Exception as e:
    print(f"Error loading prompting module (07_prompting.py): {e}")
    raise SystemExit(1)

print("Pipeline loaded.\n")


# ---------------------------------------------------------------------------
# Chat State
# ---------------------------------------------------------------------------


class ChatState:
    """Maintains state across chat turns."""
    def __init__(self):
        self.mode: str = "hybrid"  # hybrid, semantic, structured
        self.model: str = "haiku"  # haiku, sonnet, gpt4, etc.
        self.last_context: str = ""
        self.last_route: str = ""
        self.last_prompt: str = ""
        self.last_citations: list[dict] = []
        self.show_context: bool = False
        self.show_route: bool = False
        self.history: list[dict] = []  # [{"role": "user/assistant", "content": ...}]
        self.max_history: int = 10
        self.artifact_paths: ArtifactPaths | None = None
        self.memory: ConversationMemory = ConversationMemory()


state = ChatState()


def configure_runtime_dataset(
    competition_id: int = 43,
    season_id: int = 106,
    legacy_default: bool = True,
    embedding_model_id: str | None = None,
) -> None:
    """Select one dataset (and embedding model) for all queries in this chat
    runtime. `embedding_model_id=None` keeps the existing MiniLM-compatible
    default -- see src.embedding_config."""
    state.artifact_paths = resolve_runtime_artifact_paths(
        competition_id,
        season_id,
        legacy_default=legacy_default,
        embedding_model_id=embedding_model_id,
    )


# ---------------------------------------------------------------------------
# Query Processing
# ---------------------------------------------------------------------------


def process_query(question: str) -> str:
    """
    Process a question through the full RAG pipeline.

    Returns the generated answer.
    """
    # Step 0: Search dataset-scoped conversation memory for context relevant
    # to this question. This never supplies football facts -- it only
    # (a) may resolve a pronoun in the *retrieval* query below, and
    # (b) is surfaced to the LLM as labeled, non-authoritative context later.
    relevant_turns = state.memory.search(state.artifact_paths, question)
    retrieval_query = resolve_pronoun_references(question, relevant_turns)

    # Step 1: Route the query (respect mode override)
    if state.mode == "structured":
        # Force structured path — route normally but execute structured only
        route = router_mod.route_query(
            retrieval_query,
            artifact_paths=state.artifact_paths,
        )
        if route.path != "structured" or not route.structured_query:
            # Can't parse as structured — fall back to hybrid
            route.path = "hybrid"
    elif state.mode == "semantic":
        # Force semantic path — skip structured
        route = router_mod.Route(
            path="semantic",
            confidence=1.0,
            reason="User forced semantic mode",
            semantic_query=retrieval_query,
        )
    else:
        # hybrid — use normal routing
        route = router_mod.route_query(
            retrieval_query,
            artifact_paths=state.artifact_paths,
        )

    # Step 2: Execute the route
    result = router_mod.execute_route(route, semantic_k=5, artifact_paths=state.artifact_paths)

    # Store state
    state.last_route = (
        f"Path: {route.path}\n"
        f"Confidence: {route.confidence:.2f}\n"
        f"Reason: {route.reason}"
    )

    # Step 3/4: Build context (shared policy -- src/orchestration/policy.py).
    # Structured facts first and flagged authoritative, retrieved chunks after,
    # then the relevant conversation turns from Step 0 prepended as labeled,
    # reference-only context that is never merged into the authoritative
    # section. 07_prompting.py::answer_question() runs the identical policy;
    # only `semantic_k` (5 here vs 3 there) and this fallback message differ,
    # and both are genuine interface choices rather than duplication.
    sr = result.structured_result
    assembled = orchestration.assemble_context(
        result,
        conversation_context=format_conversation_context(relevant_turns),
    )
    has_structured = assembled.has_structured
    context = assembled.context
    full_context = assembled.full_context

    state.last_context = context

    prompt = prompting_mod.build_prompt(question, full_context, has_structured=has_structured)
    state.last_prompt = prompt

    # Trust boundary (security parity with 07_prompting.py::answer_question):
    # send developer policy as a `system` message and retrieved evidence +
    # question as a `user` message, so retrieved text cannot occupy the system
    # role whatever it contains. Before this, the CLI sent one combined `user`
    # message -- policy and untrusted evidence at the same privilege level,
    # separated only by markdown delimiters -- while the Streamlit path was
    # already hardened. `prompt` is still built (and still shown by /prompt)
    # and is byte-identical to these two contents joined; see
    # src/generation/prompt.py and tests/test_generation_verification.py.
    messages = prompting_mod.build_messages(
        question, full_context, has_structured=has_structured
    )

    # Step 5: Generate answer
    state.history.append({"role": "user", "content": question})
    citations: list[dict] = []
    if orchestration.should_refuse(result):
        # No citation list for the deterministic refusal, even if evidence
        # was retrieved -- it was judged insufficient, so showing it here
        # would misleadingly imply it supports an answer it does not.
        answer = prompting_mod.INSUFFICIENT_CONTEXT_MESSAGE
    else:
        try:
            answer = prompting_mod.generate_answer(
                prompt, model=state.model, messages=messages
            )
        except Exception as e:
            # CLI-only fallback: surface the retrieved context so the user can
            # still see the evidence. Deliberately leaves citations empty --
            # an error string has no sources -- which is why finalize_answer()
            # runs only on the success path below.
            answer = f"[LLM Error: {e}]\n\nRetrieved context ({len(context)} chars, showing first 2000):\n{context[:2000]}..."
        else:
            # Step 6: verification + citations (shared policy). Previously the
            # validation block sat outside this branch, but it could only ever
            # fire here: should_refuse() is true only when there is no usable
            # structured result, so `has_structured` is always False on the
            # refusal path.
            finalized = orchestration.finalize_answer(answer, result, has_structured)
            answer = finalized.answer
            citations = finalized.citations
            if finalized.corrected:
                print(f"[VALIDATION] Contradiction detected: {finalized.validation}")

    state.last_citations = citations
    state.history.append({"role": "assistant", "content": answer})
    # Trim history to max size
    if len(state.history) > state.max_history * 2:
        state.history = state.history[-state.max_history * 2:]

    state.memory.add_turn(state.artifact_paths, question, answer)

    return answer


def print_separator():
    """Print a visual separator."""
    print("\n" + "=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Interactive Mode
# ---------------------------------------------------------------------------


def interactive_mode():
    """Run interactive chat loop."""
    print("Football Analytics — RAG Chat")
    print("Type your question, or /help for commands.\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                handle_command(user_input)
                continue

            # Process question
            print("\nRetrieving and generating...")
            answer = process_query(user_input)

            # Print answer
            print_separator()
            print(f"Question: {user_input}")
            print_separator()
            print(f"Answer:\n{answer}")
            sources_block = prompting_mod.render_citations_cli(state.last_citations)
            if sources_block:
                print(f"\n{sources_block}")
            print_separator()

            # Optionally show context/route
            if state.show_context:
                print(f"Context:\n{state.last_context}")
                print_separator()
            if state.show_route:
                print(f"Route:\n{state.last_route}")
                print_separator()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\n\nGoodbye!")
            break


def single_question_mode(question: str):
    """Process a single question and exit."""
    print(f"Question: {question}")
    print_separator()

    answer = process_query(question)

    print(f"Answer:\n{answer}")
    sources_block = prompting_mod.render_citations_cli(state.last_citations)
    if sources_block:
        print(f"\n{sources_block}")
    print_separator()

    print(f"Context:\n{state.last_context}")
    print_separator()

    print(f"Route:\n{state.last_route}")


# ---------------------------------------------------------------------------
# Command Handling
# ---------------------------------------------------------------------------


def handle_command(cmd: str):
    """Handle chat commands."""
    cmd = cmd.lower().strip()

    if cmd == "/help":
        print(__doc__)

    elif cmd == "/context":
        if state.last_context:
            print(f"\nRetrieved Context:\n{state.last_context}")
        else:
            print("\nNo context available yet. Ask a question first.")

    elif cmd == "/prompt":
        if state.last_prompt:
            print(f"\nFull Prompt:\n{state.last_prompt}")
        else:
            print("\nNo prompt available yet. Ask a question first.")

    elif cmd == "/route":
        if state.last_route:
            print(f"\nRouting Decision:\n{state.last_route}")
        else:
            print("\nNo route available yet. Ask a question first.")

    elif cmd == "/mode":
        modes = ["hybrid", "semantic", "structured"]
        current = state.mode
        print(f"\nCurrent mode: {current}")
        print(f"Available modes: {', '.join(modes)}")
        new_mode = input("Switch to: ").strip().lower()
        if new_mode in modes:
            state.mode = new_mode
            print(f"Mode switched to: {new_mode}")
        else:
            print(f"Invalid mode: {new_mode}")

    elif cmd == "/model":
        models = list(prompting_mod.MODELS.keys())
        current = state.model
        print(f"\nCurrent model: {current}")
        print(f"Available models: {', '.join(models)}")
        new_model = input("Switch to: ").strip().lower()
        if new_model in prompting_mod.MODELS:
            state.model = new_model
            print(f"Model switched to: {new_model}")
        else:
            print(f"Invalid model: {new_model}")

    elif cmd == "/toggle-context":
        state.show_context = not state.show_context
        print(f"Show context after answer: {state.show_context}")

    elif cmd == "/toggle-route":
        state.show_route = not state.show_route
        print(f"Show route after answer: {state.show_route}")

    elif cmd == "/history":
        if state.history:
            print("\nConversation History:")
            for turn in state.history:
                role = "You" if turn["role"] == "user" else "Assistant"
                print(f"  {role}: {turn['content'][:150]}...")
        else:
            print("\nNo conversation history yet.")

    elif cmd == "/clear":
        state.history.clear()
        state.memory.clear(state.artifact_paths)
        print("Conversation history cleared.")

    elif cmd in ("/quit", "/exit", "/q"):
        print("Goodbye!")
        sys.exit(0)

    else:
        print(f"Unknown command: {cmd}. Type /help for commands.")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def print_datasets() -> None:
    """List runtime-selectable datasets discovered under output/."""
    entries = discover_datasets(Path("output"))
    if not entries:
        print("No datasets discovered under output/.")
        return
    for entry in entries:
        status = "ready" if entry.is_ready else "not ready"
        print(f"  competition_id={entry.competition_id} season_id={entry.season_id}  "
              f"{entry.label}  [{status}]")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Interactive football analytics RAG chat")
    parser.add_argument("--competition-id", type=int)
    parser.add_argument("--season-id", type=int)
    parser.add_argument(
        "--namespaced",
        action="store_true",
        help="Use the namespaced artifact layout for the legacy default dataset",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List discovered runtime datasets (from output/) and exit",
    )
    from src.embedding_config import DEFAULT_EMBEDDING_MODEL_ID, EMBEDDING_MODELS
    parser.add_argument(
        "--embedding-model", default=None, choices=sorted(EMBEDDING_MODELS),
        help=f"Registered embedding model id to query with (default: {DEFAULT_EMBEDDING_MODEL_ID}). "
             f"Must match the model the selected dataset's Chroma index was built with.",
    )
    parser.add_argument("question", nargs="*")
    args = parser.parse_args()

    if args.list_datasets:
        print_datasets()
        return 0

    if (args.competition_id is None) != (args.season_id is None):
        parser.error("--competition-id and --season-id must be provided together")

    if args.competition_id is None:
        configure_runtime_dataset(legacy_default=not args.namespaced,
                                   embedding_model_id=args.embedding_model)
    else:
        configure_runtime_dataset(
            args.competition_id,
            args.season_id,
            legacy_default=not args.namespaced,
            embedding_model_id=args.embedding_model,
        )

    # Check for API key
    try:
        prompting_mod.get_api_key()
    except ValueError as e:
        print(f"Warning: {e}")
        print("You can still test retrieval, but LLM generation will fail.\n")

    if args.question:
        single_question_mode(" ".join(args.question))
        return 0

    interactive_mode()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
