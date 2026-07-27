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

import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Pipeline Integration
# ---------------------------------------------------------------------------


def import_module(name: str, path: str):
    """Import a module from a file path."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load modules
print("Loading RAG pipeline...")
try:
    router_mod = import_module("router", "06_retrieve_context.py")
except Exception as e:
    print(f"Error loading router (06_retrieve_context.py): {e}")
    print("Make sure all dependencies are installed: pip install -r requirements.txt")
    raise SystemExit(1)

try:
    prompting_mod = import_module("prompting", "07_prompting.py")
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
        self.show_context: bool = False
        self.show_route: bool = False
        self.history: list[dict] = []  # [{"role": "user/assistant", "content": ...}]
        self.max_history: int = 10


state = ChatState()


# ---------------------------------------------------------------------------
# Query Processing
# ---------------------------------------------------------------------------


def process_query(question: str) -> str:
    """
    Process a question through the full RAG pipeline.

    Returns the generated answer.
    """
    # Step 1: Route the query (respect mode override)
    if state.mode == "structured":
        # Force structured path — route normally but execute structured only
        route = router_mod.route_query(question)
        if route.path != "structured" or not route.structured_query:
            # Can't parse as structured — fall back to hybrid
            route.path = "hybrid"
    elif state.mode == "semantic":
        # Force semantic path — skip structured
        route = router_mod.Route(
            path="semantic",
            confidence=1.0,
            reason="User forced semantic mode",
            semantic_query=question,
        )
    else:
        # hybrid — use normal routing
        route = router_mod.route_query(question)

    # Step 2: Execute the route
    result = router_mod.execute_route(route, semantic_k=5)

    # Store state
    state.last_route = (
        f"Path: {route.path}\n"
        f"Confidence: {route.confidence:.2f}\n"
        f"Reason: {route.reason}"
    )

    # Step 3: Build context.
    # Combine both paths so hybrid queries don't silently discard the exact
    # structured answer. The structured result is presented first and flagged
    # as authoritative; semantic chunks follow as supporting narrative.
    parts = []
    sr = result.structured_result
    has_structured = False
    if sr and sr.status in ("resolved", "partial") and sr.explanation:
        # Mark structured data as authoritative
        parts.append(
            "## Authoritative Data (Verified from Match Facts)\n\n"
            "The following numbers are VERIFIED and must be used EXACTLY:\n\n"
            + sr.explanation
        )
        has_structured = True
    if result.semantic_chunks:
        parts.append(prompting_mod.format_context_for_prompt(result.semantic_chunks))

    context = "\n\n".join(parts) if parts else "No relevant context found."

    state.last_context = context

    # Step 4: Build prompt (include conversation history for follow-up support)
    history_text = ""
    if state.history:
        recent = state.history[-state.max_history:]
        history_lines = []
        for turn in recent:
            role = "User" if turn["role"] == "user" else "Assistant"
            history_lines.append(f"{role}: {turn['content'][:200]}")
        history_text = "\n".join(history_lines)

    if history_text:
        full_context = f"## Previous conversation\n{history_text}\n\n{context}"
    else:
        full_context = context

    prompt = prompting_mod.build_prompt(question, full_context, has_structured=has_structured)
    state.last_prompt = prompt

    # Step 5: Generate answer
    state.history.append({"role": "user", "content": question})
    try:
        answer = prompting_mod.generate_answer(prompt, model=state.model)
    except Exception as e:
        answer = f"[LLM Error: {e}]\n\nRetrieved context ({len(context)} chars, showing first 2000):\n{context[:2000]}..."

    # Step 6: Validate answer against structured facts
    if sr and sr.status in ("resolved", "partial") and sr.explanation:
        try:
            from prompting import validate_answer
            validation = validate_answer(
                llm_answer=answer,
                structured_explanation=sr.explanation,
                structured_value=sr.aggregated_value,
            )
            if not validation.is_valid:
                # Contradiction detected — use corrected answer
                answer = validation.corrected_answer or answer
                print(f"[VALIDATION] Contradiction detected: {validation}")
        except Exception as e:
            # Validation is best-effort — don't break the pipeline
            pass

    state.history.append({"role": "assistant", "content": answer})
    # Trim history to max size
    if len(state.history) > state.max_history * 2:
        state.history = state.history[-state.max_history * 2:]

    return answer


def print_separator():
    """Print a visual separator."""
    print("\n" + "=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Interactive Mode
# ---------------------------------------------------------------------------


def interactive_mode():
    """Run interactive chat loop."""
    print("FIFA World Cup 2022 — RAG Chat")
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
        print("Conversation history cleared.")

    elif cmd in ("/quit", "/exit", "/q"):
        print("Goodbye!")
        sys.exit(0)

    else:
        print(f"Unknown command: {cmd}. Type /help for commands.")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def main() -> int:
    """Main entry point."""
    # Check for API key
    try:
        prompting_mod.get_api_key()
    except ValueError as e:
        print(f"Warning: {e}")
        print("You can still test retrieval, but LLM generation will fail.\n")

    # Single question mode
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        single_question_mode(question)
        return 0

    # Interactive mode
    interactive_mode()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
