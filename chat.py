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
router_mod = import_module("router", "08_router.py")
prompt_mod = import_module("prompt_builder", "src/generation/prompt_builder.py")
llm_mod = import_module("llm", "src/generation/llm.py")
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


state = ChatState()


# ---------------------------------------------------------------------------
# Query Processing
# ---------------------------------------------------------------------------


def process_query(question: str) -> str:
    """
    Process a question through the full RAG pipeline.

    Returns the generated answer.
    """
    # Step 1: Route the query
    route = router_mod.route_query(question)

    # Step 2: Execute the route
    result = router_mod.execute_route(route, semantic_k=5)

    # Store state
    state.last_route = (
        f"Path: {route.path}\n"
        f"Confidence: {route.confidence:.2f}\n"
        f"Reason: {route.reason}"
    )

    # Step 3: Build context
    if result.semantic_chunks:
        context = prompt_mod.format_context_for_prompt(result.semantic_chunks)
    elif result.structured_result and result.structured_result.explanation:
        context = result.structured_result.explanation
    else:
        context = "No relevant context found."

    state.last_context = context

    # Step 4: Build prompt
    prompt = prompt_mod.build_prompt(question, context)
    state.last_prompt = prompt

    # Step 5: Generate answer
    try:
        answer = llm_mod.generate_answer(prompt, model=state.model)
    except Exception as e:
        answer = f"[LLM Error: {e}]\n\nRetrieved context was:\n{context[:500]}..."

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
        models = list(llm_mod.MODELS.keys())
        current = state.model
        print(f"\nCurrent model: {current}")
        print(f"Available models: {', '.join(models)}")
        new_model = input("Switch to: ").strip().lower()
        if new_model in llm_mod.MODELS:
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
        llm_mod.get_api_key()
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
