"""
CLI entrypoint for the greeter agent.

Provides an interactive command-line interface for testing the agent.
The greeter agent node handles user input and printing AI responses internally.
"""

import sys
import uuid
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.graph.builder import build_graph
from langchain_core.messages import AIMessage, HumanMessage


def main():
    """Main CLI entrypoint."""
    print("\n" + "="*70)
    print("DEUS BANK - GREETER AGENT")
    print("="*70 + "\n")

    # Build the graph
    app = build_graph()

    # Generate a unique thread ID for this session
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    start_message = "Hello! Thank you for reaching out to DEUS Bank. To assist you better, could you please provide your name and either your phone number or IBAN? This will help us verify your identity.\n"
    print(start_message)

    try:
        first_run = True
        while True:
            try:
                user_input = input("You: ")
            except EOFError:
                break
                
            if user_input.strip().lower() in ["exit", "quit"]:
                break
            
            messages = [HumanMessage(content=user_input)]
            
            # On the first run, inject the AI's greeting into the history 
            # so the model knows it has already greeted the user.
            if first_run:
                messages.insert(0, AIMessage(content=start_message))
                first_run = False
            
            # Run the graph
            final_state = app.invoke({"messages": messages}, config=config)

            # Get the last message from the agent
            state_messages = final_state.get("messages", [])
            if state_messages:
                last_msg = state_messages[-1]
                if isinstance(last_msg, AIMessage):
                    print(f"\nAI: {last_msg.content}\n")
            
            # Debug output (optional, commented out for cleaner UX)
            # print("-" * 30)
            # print("CURRENT STATE:")
            # if "customer_data" in final_state and final_state["customer_data"]:
            #     print(final_state["customer_data"].model_dump_json(indent=2, exclude_none=True))
            # print("-" * 30 + "\n")

    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        # raise # Uncomment to see full traceback during development


if __name__ == "__main__":
    main()
